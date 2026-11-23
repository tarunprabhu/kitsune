//===- LocalizeReductions.cpp - Compute local reductions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compute per-thread reductions in prepared tapir loops. At the end of the
// thread, accumulate this contribution to the original destination. Consider
// the tapir reduction loop shown below:
//
//     int res = <identity>;
//     forall(...) {
//       __kitsune_reduce(&res, ...)
//     }
//
// This will be transformed to the following:
//
//     int res = <identity>;
//     forall(...) {
//       int local = <identity>;
//       __kitsune_reduce(&local, ...)
//       __kitsune_reduce(&res, local, ...);
//     }
//
// From the example above, it is not clear why this transformation is
// beneificial. However, in general, the reduction could be performed inside
// nested serial loops within the parallel loop body. In such cases, performing
// a local reduction, which can be performed in serial, followed by a parallel
// reduction into the actual result can be more efficient.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/LocalizeReductions.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/Reductions.h"
#include "kitsune/Core/TTUtils.h"
#include "kitsune/Core/ValueUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/IRBuilder.h"

#define DEBUG_TYPE "kit-reduce-localize"

using namespace llvm;

namespace {

class LocalizeReductions {
public:
  // Map from the destinations of reductions to the local into which they will
  // be reduced before performing the final reduction.
  using Locals = DenseMap<Value *, AllocaInst *>;

protected:
  LoopInfo &li;

protected:
  AllocaInst *genLocal(const ReductionInfo &redxn, Instruction *insertPt);
  Locals genLocals(Loop &loop, const Reductions &redxns);
  void genReduceIntoLocal(Loop &loop, const Reductions &redxns,
                          const Locals &locals);
  void genReduceIntoFinal(Loop &loop, const Reductions &redxn,
                          const Locals &locals);
  void updateCallAttributes(Loop &loop, const Reductions &redxns);
  bool localizeReduction(Loop &loop, const Reductions &redxns);
  bool localizeReductionCPU(Loop &loop, const Reductions &redxns);
  bool localizeReductionGPU(Loop &loop, const Reductions &redxns);
  bool localizeReductionSerial(Loop &loop, const Reductions &redxns);
  bool check(const Loop &loop, const Reductions &redxns);
  bool run(Loop &loop);

public:
  LocalizeReductions(LoopInfo &li) : li(li) {}

  bool run(Function &f);
};

} // namespace

AllocaInst *LocalizeReductions::genLocal(const ReductionInfo &redxn,
                                         Instruction *insertPt) {
  Type *type = redxn.getType();
  Value *unit = redxn.getUnit();
  unsigned size = redxn.elemSize;
  Type *localType = redxn.getResultBufferType();

  IRBuilder<> builder(insertPt);
  AllocaInst *local = builder.CreateAlloca(localType, /*ArraySize=*/nullptr);
  if (isa<PointerType>(type))
    builder.CreateMemCpy(local, MaybeAlign(), unit, MaybeAlign(), size);
  else
    builder.CreateStore(unit, local);

  return local;
}

LocalizeReductions::Locals
LocalizeReductions::genLocals(Loop &loop, const Reductions &redxns) {
  // Get the point at which to insert allocas for the local reductions. We
  // cannot use BasicBlock::getFirst* functions because the variant that looks
  // for allocas only works if the basic block is the function entry block.
  auto getInsertPoint = [](BasicBlock &bb) -> Instruction * {
    for (Instruction &inst : bb)
      if (!isa<AllocaInst>(inst) && !isa<PHINode>(inst))
        return &inst;
    llvm_unreachable("No valid insert point in basic block");
  };

  // We add the alloca to the detached block of the tapir loop. This is
  // essentially the "entry block" of the loop body. These tapir loops will
  // eventually be outlined. Tapir loops with a CPU-centric tapir target are
  // guaranteed to be of depth 1, with exactly one iteration of the tapir loop
  // executing per thread. As a result, any allocas that are in the body of the
  // loop will end up in the function entry block after outlining. On GPU's, the
  // loop whose reductions are localized will be the innermost loop in the nest.
  // There, too, exactly one iteration of the nest is executed per GPU thread,
  // so the alloca will again end up in the entry block of the kernel function.
  // We don't create a local alloca for each reduction, but for each unique
  // value that is used as the destination of a reduction. This ensures that we
  // have exactly as many local allocas as there are unique destinations of
  // reductions.
  DetachInst *detach = getUniqueInstInLoopOnly<DetachInst>(loop);
  BasicBlock *bbBody = detach->getDetached();

  Locals locals;
  for (const ReductionInfo &redxn : redxns) {
    Value *dest = redxn.getDest();
    if (!locals.contains(dest))
      locals[dest] = genLocal(redxn, getInsertPoint(*bbBody));
  }
  return locals;
}

void LocalizeReductions::genReduceIntoLocal(Loop &loop,
                                            const Reductions &redxns,
                                            const Locals &locals) {
  // Since we are guaranteed that the destination of the reduction is not used
  // elsewhere in the loop (except perhaps as the destination of other
  // reductions in the loop), to reduce into the local instead, we only need to
  // replace the destination in the call to the reduce intrinsic. We also update
  // the reduction strategy to naive since this is always a straightforward
  // serial reduction.
  LLVMContext &ctx = getContext(loop);
  Value *strategy = toConstant(ReductionStrategy::Naive, ctx);

  for (const ReductionInfo &redxn : redxns) {
    AllocaInst *local = locals.at(redxn.getDest());

    redxn.call->setArgOperand(ReductionInfo::argNumDest, local);
    redxn.call->setArgOperand(ReductionInfo::argNumStrategy, strategy);
  }
}

void LocalizeReductions::genReduceIntoFinal(Loop &loop,
                                            const Reductions &redxns,
                                            const Locals &locals) {
  // The reattach instruction is guaranteed to be the very last instruction in
  // the loop body. Once outlined, any instructions added right before the
  // reattach instruction will effectively be the very last instructions in the
  // outlined function, except for the return. The accumulation into the final
  // destination can be safely performed there where it is guaranteed to be
  // executed after the local reduction has been computed.
  ReattachInst *reattach = getUniqueInstInLoopOnly<ReattachInst>(loop);
  IRBuilder<> builder(reattach);

  // We first insert the loads of the local. This is so the calls to the reduce
  // intrinsics stay grouped together. This can be useful for a future
  // optimization that fuses the final reductions.
  DenseMap<Value *, LoadInst *> loads;
  for (const ReductionInfo &redxn : redxns) {
    Type *type = redxn.getType();
    Value *dest = redxn.getDest();
    AllocaInst *local = locals.at(dest);
    LoadInst *load = builder.CreateLoad(type, local);
    loads[dest] = load;
  }

  // Now add the final reduction call.
  for (const ReductionInfo &redxn : redxns) {
    Value *tt = redxn.getTTV();
    Value *op = redxn.getReduceOpV();
    Value *dest = redxn.getDest();
    Value *size = redxn.getElemSizeV();
    Value *unit = redxn.getUnit();
    Value *reducer = redxn.getReducer();
    Value *strategy = redxn.getStrategyV();
    Value *value = loads.at(dest);

    SmallVector<Type *, 2> overloadTys = redxn.getOverloadTypes();
    SmallVector<Value *, 8> args = {tt,       op,   dest, value,
                                    strategy, size, unit, reducer};
    args.append(redxn.getExtraArgs());
    builder.CreateIntrinsic(Intrinsic::kit_reduce, overloadTys, args);
  }
}

void LocalizeReductions::updateCallAttributes(Loop &loop,
                                              const Reductions &redxns) {
  for (const ReductionInfo &redxn : redxns) {
    // The original reduction call may have been marked as tail since the
    // destination of the reduction may be not be a stack pointer. However,
    // after reducing into an alloca, it is no longer safe for this call to be
    // a tail call.
    redxn.call->setTailCall(CallInst::TCK_None);
  }
}

bool LocalizeReductions::localizeReduction(Loop &loop,
                                           const Reductions &redxns) {
  // Generate allocas for the local reduction. We generate one alloca per
  // destination - not one alloca per reduction.
  Locals locals = genLocals(loop, redxns);

  // Generate the final reduction. We do this first because we need the original
  // destination, which will eventually be replaced with the thread-local stack
  // slot that was allocated for it.
  genReduceIntoFinal(loop, redxns, locals);

  // It is now safe to update the original reduction to reduce into the local
  // variable that was allocated. This will also update the reduction strategy.
  // After this, it is no longer safe to use the ReductionInfo object.
  genReduceIntoLocal(loop, redxns, locals);

  // Update original reduction attributes.
  updateCallAttributes(loop, redxns);

  return true;
}

bool LocalizeReductions::localizeReductionCPU(Loop &loop,
                                              const Reductions &redxns) {
  bool changed = localizeReduction(loop, redxns);
  addLocalizedAttr(loop);
  return changed;
}

bool LocalizeReductions::localizeReductionGPU(Loop &root,
                                              const Reductions &redxns) {
  SmallVector<Loop *, 0> loops = getTapirLoopsInNest(root);
  bool changed = localizeReduction(*loops.back(), redxns);
  for (Loop *loop : loops)
    addLocalizedAttr(*loop);
  return changed;
}

bool LocalizeReductions::localizeReductionSerial(Loop &loop,
                                                 const Reductions &redxns) {
  // There is nothing to be gained by localizing reductions for loops that will
  // eventually be executed in serial.
  return false;
}

bool LocalizeReductions::check(const Loop &loop, const Reductions &redxns) {
  if (!loop.isLoopSimplifyForm())
    return emitDiagnostic(loop, DiagID::ErrLoopNotSimplifyForm);

  for (const ReductionInfo &redxn : redxns)
    if (std::optional<DiagID> err = redxn.checkUsesOfDest(li))
      return emitDiagnostic(loop, *err, getName(*redxn.getDest()));

  return true;
}

bool LocalizeReductions::run(Loop &loop) {
  assert(isTapirLoop(loop) && "Loop is a tapir loop");
  assert(hasReductionAttr(loop) && "Loop is a reduction loop");
  assert(getUniqueInstInLoopOnly<DetachInst>(loop) &&
         "Tapir loop has a unique detach instruction");
  assert(getUniqueInstInLoopOnly<ReattachInst>(loop) &&
         "Tapir loop has a unique reattach instruction");

  Reductions redxns = getReductions(loop);
  if (!check(loop, redxns))
    return false;

  TTID tt = *getTargetAttr(loop);
  if (tt == TTID::Serial)
    return localizeReductionSerial(loop, redxns);
  else if (isCPUTT(tt))
    return localizeReductionCPU(loop, redxns);
  else if (isGPUTT(tt))
    return localizeReductionGPU(loop, redxns);
  else if (tt == TTID::Custom)
    return false;
  else
    llvm_unreachable("Unknown tapir target category");
}

bool LocalizeReductions::run(Function &f) {
  bool changed = false;
  for (Loop *loop : getTopLevelTapirLoops(li))
    if (hasReductionAttr(*loop) && !hasLocalizedAttr(*loop))
      changed |= run(*loop);
  return changed;
}

PreservedAnalyses LocalizeReductionsPass::run(Function &f,
                                              FunctionAnalysisManager &am) {
  LoopInfo &li = am.getResult<LoopAnalysis>(f);

  if (LocalizeReductions(li).run(f))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
