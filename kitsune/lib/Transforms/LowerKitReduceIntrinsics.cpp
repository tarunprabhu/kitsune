//===- LowerKitReduceIntrinsics.cpp - Lower Kitsune's reduce intrinsics ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune's reduce intrinsics.
//
// Unlike most other intrinsics, this is done as part of the middle end. The
// reduction may involve generation of LLVM loops, or even tapir loops.
// Typically, several optimization passes will have to be run after this pass
// to ensure that any code that is generated here is optimized.
//
//
// kit.reduce.0 is reduced by simply calling the provided reducer function.
// For example, the call below
//
//   call void kit.reduce.0(i32 1, ptr %r, i32 4, i32 %v, i32 1, ptr @f, i8 %e)
//
// would become
//
//   call void @f(ptr %r, i32 %v, i8 %e)
//
// Note that here, %e is part of the optional arguments that the reducer
// function might accept.
//
//
// kit.reduce.1 is reduced in two steps. In the first step, each call to
// the function is replaced with a straightforward loop, where each iteration of
// the loop calls kit.reduce.0 to with one element of the array. Consider the
// call below:
//
//   call void kit.reduce.1(i32 1, ptr %r, i32 8, ptr %a, i64 %n, i64 0, ptr @f)
//
// The following high-level pseudocode shows the first stage of this lowering.
//
//   for (size_t i = 0; i < %n; ++i) {
//       %v = %a[%i]
//       call void kit.reduce.0(i32 1, ptr %r, i32 8, i64 %v, i64 0, ptr @f)
//   }
//
// If successive calls to kit.reduce.1 are found, a fused loop is generated.
// The following constraints must be satisfied for this fusion to take place:
//
//   - The calls must be in the same block and there are non non-debug
//     instructions between the calls.
//
//   - The TTID argument is the same on the calls, as is the number of elements
//     on which the two calls operate.
//
// The code below shows an example of successive calls to kit.reduce.1 and
// pseudo-code for the fused loop. Note that here, the reducer functions and
// the element sizes are different in the two calls. But they were, nevertheless
// fused
//
//   call void kit.reduce.1(i32 1, ptr %r, i32 8, ptr %a, i64 %n, i64 0, ptr @f)
//   call void kit.reduce.1(i32 1, ptr %r, i32 4, ptr %b, i64 %n, i32 1, ptr @g)
//
//   for (size_t i = 0; i < %n; ++i) {
//       %v1 = %a[%i]
//       call void kit.reduce.0(i32 1, ptr %r, i32 8, i64 %v1, i64 0, ptr @f)
//       %v2 = %b[%i]
//       call void kit.reduce.0(i32 1, ptr %r, i32 4, i64 %v1, i32 1, ptr @g)
//   }
//
// In the second step, the calls to kit.reduce.0 are lowered as described
// earlier.
//
// ---------------------------------- WARNING ----------------------------------
//
// Merging successive calls to the kit.reduce.1 intrinsic is only valid if the
// destinations of the reductions and the sources are disjoint. However, this
// pass does not check that this is actually the case, and fuses the calls
// unconditionally. This is generally safe only because we expect the
// kit.reduce.1 intrinsics to have been inserted by the kit-prepare pass
// when lowering tapir reduction loops. In those cases, we are guaranteed to be
// alias-free. If there are aliases, the semantics of tapir reduction loops will
// have been violated - in which case, we are absolved of the guilt for this
// miscompilation.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/LowerKitReduceIntrinsics.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/Reductions.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "kit-reduce-intrinsics"

using namespace llvm;

namespace {

using Calls = SmallVector<CallBase *, 0>;

// A group of successive calls to the kit.reduce.1 intrinsic. All calls in the
// group must satisfy *all* of the following conditions:
//
//   - The calls must be consecutive in the _same_ basic block. There must be
//     no non-debug instructions between successive calls.
//
//   - All calls must have the same TTID argument.
//
//   - All calls must operate on the same number of array elements. The sizes
//     of the array elements do not matter
//
struct ReductionGroup {
  Value *tt = nullptr;
  Value *numElems = nullptr;
  Calls calls;

  ReductionGroup(CallBase *call)
      : tt(call->getArgOperand(0)), numElems(call->getArgOperand(4)),
        calls({call}) {}

  bool empty() const { return calls.empty(); }
  size_t size() const { return calls.size(); }
  CallBase *front() const { return calls.front(); }
  CallBase *back() const { return calls.back(); }
  Calls::const_iterator begin() const { return calls.begin(); }
  Calls::const_iterator end() const { return calls.end(); }

  void add(CallBase *call) { calls.push_back(call); }
};

// Lower calls kit.reduce.1 intrinsics.
class LowerReduce1 {
private:
  using ReductionGroups = SmallVector<ReductionGroup, 1>;

private:
  DominatorTree &dt;
  LoopInfo &li;

  DomTreeUpdater dtu;
  MemorySSAUpdater mssau;

private:
  // Get arguments from calls to the kit.reduce.1 intrinsic.
  Value *getTTID(const CallBase &call) const { return call.getArgOperand(0); }
  Value *getDest(const CallBase &call) const { return call.getArgOperand(1); }
  Value *getElemSize(const CallBase &c) const { return c.getArgOperand(2); }
  Value *getElems(const CallBase &call) const { return call.getArgOperand(3); }
  Value *getNumElems(const CallBase &c) const { return c.getArgOperand(4); }
  Value *getUnit(const CallBase &call) const { return call.getArgOperand(5); }
  Value *getReducer(const CallBase &c) const { return c.getArgOperand(6); }
  SmallVector<Value *, 0> getExtraReducerArgs(const CallBase &call) const {
    SmallVector<Value *, 0> args;
    for (unsigned i = 7; i < call.arg_size(); ++i)
      args.push_back(call.getArgOperand(i));
    return args;
  }

  ReductionGroups groupCalls(const Calls &calls);
  BasicBlock *createBodyBlock(const ReductionGroup &group);
  void createBackedgeInBody(BasicBlock &bb, const ReductionGroup &group);
  void createLoopObjectForBody(BasicBlock &bb);
  void replaceCalls(BasicBlock &bb, const ReductionGroup &group);
  void lowerGroup(const ReductionGroup &group);

public:
  LowerReduce1(DominatorTree &dt, LoopInfo &li, MemorySSA &mssa)
      : dt(dt), li(li), dtu(dt, DomTreeUpdater::UpdateStrategy::Eager),
        mssau(&mssa) {}

  // Lower call calls to kit.reduce.1 intrinsics in a function. Return true if
  // at least one call was replaced, false otherwise.
  bool run(Function &f);
};

} // namespace

// Collect all calls to the intrinsic \p id in a function.
static Calls collectCalls(Function &f, Intrinsic::ID id) {
  Calls calls;
  for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
    if (auto *call = dyn_cast<CallBase>(&*i))
      if (call->getIntrinsicID() == id)
        calls.push_back(call);
  return calls;
}

//-------------------------------- LowerReduce1 --------------------------------

// The call instructions in a group are guaranteed to be consecutive in the
// same basic block. Split the block so we are left with only these calls and a
// terminator in the block. This block will be returned.
//
// In the example below, the code on the left will be transformed to the code on
// the right.
//
//     orig:                          orig:
//       inst1                          inst1
//       inst2                          inst2
//       call kit.reduce.1              br calls
//       call kit.reduce.1
//       call kit.reduce.1            calls:
//       inst6                          call kit.reduce.1
//       inst7                          call kit.reduce.1
//       br label next                  call kit.reduce.1
//                                      br orig.rest
//
//                                    orig.rest:
//                                      inst6
//                                      inst7
//                                      br label next
//
BasicBlock *LowerReduce1::createBodyBlock(const ReductionGroup &group) {
  assert(!group.empty() && "Reduction group cannot be empty");
  assert(group.front()->getParent() == group.back()->getParent() &&
         "First and last call in the group must be in the same basic block");

  CallBase *call0 = group.front();
  CallBase *callN = group.back();
  BasicBlock *bb = call0->getParent();
  BasicBlock *bbCalls = SplitBlock(bb, call0, &dtu, &li, &mssau,
                                   "lower.reduc.1", /*Before=*/false);
  (void)SplitBlock(bbCalls, callN->getNextNode(), &dtu, &li, &mssau,
                   bb->getName() + ".rest", /*Before=*/false);
  return bbCalls;
}

// At this point, the basic block \p bb will contain only the call instructions
// to the reduce intrinsic. It will have a single predecessor and a single
// successor. Since the transformation will not introduce any additional control
// flow, the block can be treated as the header, body, and latch of a loop. This
// loop would also be in simplify form since it already has a preheader, a
// unique backedge and a dedicated exit.
//
// This will also create a new loop in the LoopInfo object.
void LowerReduce1::createBackedgeInBody(BasicBlock &bb,
                                        const ReductionGroup &group) {
  auto sanityCheck = [](const BasicBlock &bb) {
    assert(pred_size(&bb) == 1 &&
           "Block containing loop calls must have a single predecessor");
    assert(succ_size(&bb) == 1 &&
           "Block containing loop calls must have a single successor");
  };

  LLVM_DEBUG(dbgs() << "LowerKitReduce: Create loop around group of calls");
  sanityCheck(bb);

  LLVMContext &ctx = bb.getContext();
  IRBuilder<> builder(ctx);

  Type *i64 = Type::getInt64Ty(ctx);
  Constant *zero = builder.getInt64(0);
  Constant *one = builder.getInt64(1);
  BasicBlock *ph = *pred_begin(&bb);
  BasicBlock *exit = *succ_begin(&bb);
  Instruction *term = bb.getTerminator();

  builder.SetInsertPoint(bb.begin());
  PHINode *iv = builder.CreatePHI(i64, /*NumReserved=*/2, "lower.reduc.1.iv");
  iv->addIncoming(zero, ph);

  builder.SetInsertPoint(term);
  Value *inc = builder.CreateAdd(iv, one, "lower.reduc.1.iv.inc");
  Value *cmp = builder.CreateICmpEQ(inc, group.numElems, "lower.reduc.iv.cmp");

  (void)builder.CreateCondBr(cmp, exit, &bb);
  iv->addIncoming(inc, &bb);
  dt.insertEdge(&bb, &bb);

  term->eraseFromParent();

#ifndef NDEBUG
  dt.verify();
#endif // NDEBUG
}

// This should be called after the a backedge has been added from \p bb to
// itself. This will create a Loop object for this newly created loop and update
// the analyses accordingly.
void LowerReduce1::createLoopObjectForBody(BasicBlock &bb) {
  LLVM_DEBUG(dbgs() << "LowerKitReduce: Create loop object\n");

  Loop *newLoop = li.AllocateLoop();
  if (Loop *parentLoop = li.getLoopFor(&bb))
    parentLoop->addChildLoop(newLoop);
  else
    li.addTopLevelLoop(newLoop);
  newLoop->addBasicBlockToLoop(&bb, li);

#ifndef NDEBUG
  li.verify(dt);
  assert(newLoop->isLoopSimplifyForm() &&
         "Newly created loop must be in loop simplify form");
#endif // NDEBUG
}

void LowerReduce1::replaceCalls(BasicBlock &bb, const ReductionGroup &group) {
  auto sanityCheck = [](BasicBlock &bb, const ReductionGroup &group) {
    assert(group.size() && "Reduction group must not be empty");
    assert(bb.size() && "Reduction group block must not be empty");
    assert(isa<PHINode>(bb.front()) &&
           "Reduction group block must start with a phi node");
    assert(group.front()->getParent() == &bb &&
           "Replacement calls for reduction group must be inserted into the "
           "same group");
  };

  LLVM_DEBUG(dbgs() << "LowerKitReduce: Replacing calls to llvm.kit.reduce.1 "
                       "with llvm.kit.reduce.0\n");
  sanityCheck(bb, group);

  LLVMContext &ctx = bb.getContext();
  IRBuilder<> builder(&*bb.getFirstNonPHIIt());

  // Insert the replacement intrinsics first.
  for (CallBase *call : group) {
    Value *tt = group.tt;
    Value *dest = getDest(*call);
    Value *size = getElemSize(*call);
    Value *elems = getElems(*call);
    Value *unit = getUnit(*call);
    Value *reducer = getReducer(*call);
    Value *reduceOp = toConstant((uint32_t)ReduceOp::Custom, ctx);

    // The unit value is expected to be the same as the type of the elements
    // being reduced.
    Type *elemTy = unit->getType();
    PHINode *iv = cast<PHINode>(&bb.front());
    Value *addr = builder.CreateInBoundsGEP(elemTy, elems, iv);
    Value *v = builder.CreateLoad(elemTy, addr);

    Type *overloadTys[] = {elemTy, elemTy};
    SmallVector<Value *, 4> args = {tt, reduceOp, dest, size, v, unit, reducer};
    for (Value *extraArg : getExtraReducerArgs(*call))
      args.push_back(extraArg);
    CallInst *newCall =
        builder.CreateIntrinsic(Intrinsic::kit_reduce_0, overloadTys, args);
    newCall->copyMetadata(*call);
  }

  // Then remove the calls to kit.reduce.1
  for (CallBase *call : group)
    call->eraseFromParent();
}

void LowerReduce1::lowerGroup(const ReductionGroup &group) {
  LLVM_DEBUG(dbgs() << "LowerKitReduce: Lowering reduction group\n");

  BasicBlock *body = createBodyBlock(group);
  createBackedgeInBody(*body, group);
  createLoopObjectForBody(*body);
  replaceCalls(*body, group);

  LLVM_DEBUG(dbgs() << "LowerKitReduce: Done lowering reduction group\n");
}

// Group consecutive calls to kit.reduce.1 together. The calls must be
// successive instructions in the same basic block, have the same TTID, and
// operate on the same number of elements. The size of the elements need not be
// the same.
//
// Note that the last of these checks is fairly unsophisticated and simply
// relies on the LLVM values being the same. We could try to make this more
// sophisticated by allowing for equivalence, but it is not clear if there is
// any benefit to doing so
LowerReduce1::ReductionGroups LowerReduce1::groupCalls(const Calls &calls) {
  auto isImmediatelyAfter = [](Instruction &inst, Instruction &prev) -> bool {
    // FIXME: Skip any non-debug instructions between the last call in this
    // group and \p call.
    return prev.getNextNode() == &inst;
  };

  auto matches = [this](CallBase &c1, CallBase &c2) -> bool {
    return (this->getTTID(c1) == this->getTTID(c2)) &&
           (this->getNumElems(c1) == this->getNumElems(c2));
  };

  LLVM_DEBUG(dbgs() << "LowerKitReduce: Collecting reduction groups\n");
  assert(!calls.empty() && "Should have at least one call");

  ReductionGroups groups;
  groups.emplace_back(calls.front());

  for (unsigned i = 1; i < calls.size(); ++i) {
    CallBase *call = calls[i];
    ReductionGroup &group = groups.back();
    CallBase *prev = group.back();

    if (isImmediatelyAfter(*call, *prev) && matches(*call, *prev))
      group.add(call);
    else
      groups.emplace_back(call);
  }
  LLVM_DEBUG(dbgs() << "LowerKitReduce: Collected " << groups.size()
                    << " reduction groups\n");
  return groups;
}

bool LowerReduce1::run(Function &f) {
  // The cleaner way to do this would be to lower each call to kit.reduce.1,
  // then fuse the resulting loops. However, LLVM's loop-fuse pass is fairly
  // heavy, and we don't have as much control over it as we would like.
  // Ideally, we would like to be able to force two loops to be used, but that
  // is not currently possible. Instead, we do this "the hard way".
  Calls calls = collectCalls(f, Intrinsic::kit_reduce_1);
  LLVM_DEBUG(dbgs() << "LowerKitReduce: Found " << calls.size()
                    << " calls to kit.reduce.1\n");

  if (calls.size()) {
    ReductionGroups groups = groupCalls(calls);
    for (const ReductionGroup &group : groups)
      lowerGroup(group);
  }
  return calls.size();
}

//-------------------------------- LowerReduce0 --------------------------------

// The kit.reduce.0 intrinsics are replaced with a call to the reducer that
// was passed to it.
static bool lowerReduce0(Function &f) {
  LLVMContext &ctx = f.getContext();
  bool changed = false;
  for (CallBase *call : collectCalls(f, Intrinsic::kit_reduce_0)) {
    Value *res = call->getArgOperand(2);
    Value *val = call->getArgOperand(4);
    Value *reducer = call->getArgOperand(6);

    SmallVector<Value *, 4> args = {res, val};
    for (unsigned i = 7; i < call->arg_size(); ++i)
      args.push_back(call->getArgOperand(i));

    Type *voidTy = Type::getVoidTy(ctx);
    SmallVector<Type *, 4> paramTys(args.size(), nullptr);
    for (unsigned i = 0; i < args.size(); ++i)
      paramTys[i] = args[i]->getType();

    FunctionType *fty = FunctionType::get(voidTy, paramTys, /*isVarArg=*/false);
    CallInst *newCall = CallInst::Create(fty, reducer, args);
    ReplaceInstWithInst(call, newCall);

    changed = true;
  }

  return changed;
}

//------------------------ LowerKitReduceIntrinsicsPass ------------------------

PreservedAnalyses
LowerKitReduceIntrinsicsPass::run(Function &f, FunctionAnalysisManager &am) {
  DominatorTree &dt = am.getResult<DominatorTreeAnalysis>(f);
  LoopInfo &li = am.getResult<LoopAnalysis>(f);
  MemorySSA &mssa = am.getResult<MemorySSAAnalysis>(f).getMSSA();

  bool changed = false;

  // Lower kit.reduce.1 first. The lowering may introduce calls to kit.reduce.0.
  changed |= LowerReduce1(dt, li, mssa).run(f);

  // This should be done last. Although we only support 1D and 0D reduce
  // intrinsics, we may support higher-dimensional intrinsics in the future.
  // This is, essentially, the base case.
  changed |= lowerReduce0(f);

  if (changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
