//===- PrepareTapirLoopsCPU.cpp - Transform tapir loops for CPU -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform a tapir loop nest for execution on a CPU. Nests that perform
// reductions are transformed in exactly the same way as those that don't. The
// only difference is that loop nests with the serial tapir target are
// transformed differently compared to nests with those intended for parallel
// execution.
//
//===----------------------------------------------------------------------===//

#include "LoopWrapping.h"
#include "PrepareTapirLoopsImpl.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/IRBuilderUtils.h"
#include "kitsune/Core/InstUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/ValueUtils.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#define DEBUG_TYPE "kit-prepare"

using namespace llvm;

// Serialize the given loop. This will also remove the corresponding syncregion
// and sync instruction if those are the only remaining uses of the synregion.
static void serializeLoop(Loop &loop, DominatorTree &dt) {
  Value *sr = getTapirLoopSyncRegion(loop);
  serializeTapirLoop(loop, /*addSerializedAttr=*/false, &dt);

  // Since this is a nested tapir loop, it is likely that the syncregion and
  // sync instruction associated with it are dead. We might as well clean them
  // up now instead of relying on a later pass. But we make sure that it is
  // safe, just in case.
  if (sr->getNumUses() == 1 && isa<SyncInst>(*sr->user_begin())) {
    SyncInst *sync = cast<SyncInst>(*sr->user_begin());
    BranchInst *br = BranchInst::Create(sync->getSuccessor(0));
    ReplaceInstWithInst(sync, br);
    cast<Instruction>(sr)->eraseFromParent();
  }
}

namespace {

// Transform a tapir loop nest for parallel execution. Consider a tapir loop
// nest of depth 3 as shown below:
//
//     parallel_for (i1 : [0, n1])
//         parallel_for (i2 : [0, n2])
//             parallel_for (i3 : [0, n3])
//                 ...
//
// This will be transformed to the following:
//
//     sysThrds = @kit.cpu.num.threads()
//     ctx = __kitrt_nest_begin(sysThrds, 3)
//     threads = llvm.kit.nest.threads(ctx)
//     parallel_for (t : [0, threads]) {
//         [s1, e1] = __kitrt_nest_bounds(ctx, t, 0, n1, 1)
//         for (i1 : [s1, e1]) {
//             [s2, e2] = __kitrt_nest_bounds(ctx, t, 0, n2, 2)
//             for (i2 : [s2, e2]) {
//                 [s3, e3] = __kitrt_nest_bounds(ctx, t, 0, n3, 3)
//                 for (i3 : [s3, e3]) {
//                     ...
//                 }
//             }
//         }
//     }
//     llvm.kit.nest.end(ctx)
//
// Essentially, this wraps a parallel loop around the original loop nest. This
// outer loop iterates over the total number of threads available at runtime.
// The original parallel loops are all serialized.
//
class PrepareTapirNestParallel {
protected:
  DominatorTree &dt;
  LoopInfo &li;
  MemorySSA &mssa;
  ScalarEvolution &se;
  TaskInfo &ti;

  DomTreeUpdater dtu;
  MemorySSAUpdater mssau;

protected:
  Value *wrapLoopWithContext(Loop &root);
  void updateWrapperBounds(Loop &wrapperLoop, Value *outerTC);
  void updateLoopBounds(Loop &wrapperLoop, Loop &loop, Value *ctx);

public:
  PrepareTapirNestParallel(DominatorTree &dt, LoopInfo &li, MemorySSA &mssa,
                           ScalarEvolution &se, TaskInfo &ti);
  bool run(TapirLoopInfo &tapirLoop);
};

} // namespace

PrepareTapirNestParallel::PrepareTapirNestParallel(DominatorTree &dt,
                                                   LoopInfo &li,
                                                   MemorySSA &mssa,
                                                   ScalarEvolution &se,
                                                   TaskInfo &ti)
    : dt(dt), li(li), mssa(mssa), se(se), ti(ti),
      dtu(dt, DomTreeUpdater::UpdateStrategy::Eager), mssau(&mssa) {}

// Create a loop nest context object for the nest being prepared. The context
// will be created in the loop guard block and deleted in the loop bypass block.
//
// A loop nest of depth 3 as shown below
//
//     parallel_for (t : [0, n1])
//       parallel_for (i1 : [0, n1])
//         parallel_for (i2 : [0, n2])
//           parallel_for (i3 : [0, n3])
//             ...
//
// will be transformed as shown
//
//     totalThreads = call @llvm.kit.cpu.num.threads()
//     ctx = __kitrt_nest_begin(totalThreads, 3)
//     parallel_for (t : [0, n1])
//       parallel_for (i1 : [0, n1])
//         parallel_for (i2 : [0, n2])
//           parallel_for (i3 : [0, n3])
//           ...
//     __kitrt_nest_end(ctx)
//
// Note that we have inserted runtime calls directly. There are no intrinsics
// for these calls because it is not clear that they can be used for any
// optimizations.
//
// Returns the context object that was created .
Value *PrepareTapirNestParallel::wrapLoopWithContext(Loop &wrapperLoop) {
  auto sanityCheck = [](Loop &wrapperLoop) -> void {
    assert(getLoopGuardBlock(wrapperLoop, true) && "Loop has a guard block");
    assert(getLoopBypassBlock(wrapperLoop, true) && "Loop has a bypass block");
    assert(hasPerfectDepthAttr(wrapperLoop) && "Loop has depth attribute");
  };

  LLVM_DEBUG(dbgs() << "PrepareTapirNestParallel: Create loop context\n");
  sanityCheck(wrapperLoop);

  LLVMContext &ctx = getContext(wrapperLoop);
  uint32_t depth = *getPerfectDepthAttr(wrapperLoop);
  TTID tt = *getTargetAttr(wrapperLoop);

  // The depth recorded in the wrapper loop will be 1 greater than the actual
  // depth since it will include the wrapper itself.
  Constant *cdepth = toConstant(depth - 1, ctx);
  Constant *ctt = toConstant(tt, ctx);

  BasicBlock &guard =
      *getLoopGuardBlock(wrapperLoop, /*allowDeadEndExits=*/true);
  BasicBlock &bypass =
      *getLoopBypassBlock(wrapperLoop, /*allowDeadEndExits=*/true);

  IRBuilder<> builder(ctx);

  builder.SetInsertPoint(&guard.front());
  Value *thrds = builder.CreateIntrinsic(Intrinsic::kit_cpu_num_threads, {ctt},
                                         /*FMFSource=*/{}, "prlll.systhrds");
  Value *nestCtx = createCall(builder, KitFunc::kitrt_nest_begin,
                              {thrds, cdepth}, "nest.ctx");

  builder.SetInsertPoint(&bypass.front());
  createCall(builder, KitFunc::kitrt_nest_end, {nestCtx});

  return nestCtx;
}

// When the outer loop was generated, the induction variable was canonical and
// in the range [0, n] in steps of 1 where `n` was the trip count of the tapir
// loop being transformed. This must be changed so the range of the IV is
// [0, numThreads] where numThreads is the number of CPU threads available. The
// step will remain 1.
//
// When this function is called, the prepared loop nest will be as shown:
//
//     totalThreads = call @llvm.kit.cpu.num.threads()
//     ctx = __kitrt_nest_begin(totalThreads, 3)
//     parallel_for (t : [0, n1])
//       parallel_for (i1 : [0, n1])
//         parallel_for (i2 : [0, n2])
//           parallel_for (i3 : [0, n3])
//           ...
//     __kitrt_nest_end(ctx)
//
// This will transform this to the following:
//
//     totalThreads = call @llvm.kit.cpu.num.threads()
//     ctx = __kitrt_nest_begin(totalThreads, 3)
//     usedThreads = __kitrt_nest_threads(ctx)
//     parallel_for (t : [0, usedThreads])
//       parallel_for (i1 : [0, n1])
//         parallel_for (i2 : [0, n2])
//           parallel_for (i3 : [0, n3])
//           ...
//     __kitrt_nest_end(ctx)
//
void PrepareTapirNestParallel::updateWrapperBounds(Loop &wrapperLoop,
                                                   Value *ctx) {
  auto sanityCheck = [](const Loop &wrapperLoop, Value *ctx,
                        ScalarEvolution &se) {
    assert(wrapperLoop.getCanonicalInductionVariable() &&
           "Wrapper loop has a canonical induction variable");
    assert(wrapperLoop.getLoopLatch() && "Wrapper loop has a unique latch");
    assert(wrapperLoop.getLatchCmpInst() &&
           "Wrapper loop has a unique compare instruction");

    [[maybe_unused]] std::optional<Loop::LoopBounds> bounds =
        wrapperLoop.getBounds(se);
    assert(bounds.has_value() && "Wrapper loop has computable bounds");

    [[maybe_unused]] PHINode *iv = wrapperLoop.getCanonicalInductionVariable();
    [[maybe_unused]] Instruction *step = &bounds->getStepInst();
    assert(step->getOperand(0) == iv &&
           "First operand of step instruction is indction variable");
    assert(isIntOne(step->getOperand(1)) &&
           "Second operand of step instruction is 1");

    [[maybe_unused]] ICmpInst *cmp = wrapperLoop.getLatchCmpInst();
    assert(cmp->getPredicate() == CmpInst::ICMP_EQ &&
           "Predicate of compare instruction is EQ");
    assert(cmp->getOperand(0) == step &&
           "First operand of compare instruction is the step");
    assert(isZero(cmp->getOperand(1)) &&
           "Second argument of compare instruction is 0");

    assert(isa<Instruction>(ctx) && "Loop nest context is an instruction");
  };

  LLVM_DEBUG(
      dbgs()
      << "PrepareTapirNestParallel: Update outer loop induction variable\n");
  sanityCheck(wrapperLoop, ctx, se);

  IRBuilder<> builder(cast<Instruction>(ctx)->getNextNode());
  Value *thrds =
      createCall(builder, KitFunc::kitrt_nest_threads, {ctx}, "prlll.threads");

  Loop::LoopBounds bounds = *wrapperLoop.getBounds(se);
  PHINode *iv = wrapperLoop.getCanonicalInductionVariable();
  ICmpInst *cmp = wrapperLoop.getLatchCmpInst();
  Instruction *inc = &bounds.getStepInst();

  replaceNonMatchingOperands(*cmp, inc, thrds);

  // Update analyses that may have been invalidated. We can't recalculate SE,
  // but we can force it recompute the analyses for certain variables and loops
  // when they are next requested.
  se.forgetValue(iv);
  se.forgetValue(cmp);
  se.forgetLoop(&wrapperLoop);

#ifndef NDEBUG
  se.verify();
#endif // NDEBUG
}

// For CPU's, a loop of the form:
//
//     parallel_for (i in [0, n])
//       ...
//
// will have been transformed into the following before this method is called:
//
//     totalThreads = call @llvm.kit.cpu.num.threads()
//     ctx = __kitrt_nest_begin(totalThreads, 3)
//     usedThreads = __kitrt_nest_threads(ctx)
//     parallel_for (t : [0, usedThreads])
//       parallel_for (i : [0, n])
//         ...
//     __kitrt_nest_end(ctx)
//
// This will turns this to the following:
//
//     TYPE beg;
//     TYPE end;
//     ...
//
//     totalThreads = call @llvm.kit.cpu.num.threads()
//     ctx = __kitrt_nest_begin(totalThreads, 3)
//     usedThreads = __kitrt_nest_threads(ctx)
//     parallel_for (t : [0, usedThreads]) {
//       __kitrt_nest_bounds(ctx, t, 0, n, 1, &beg, &end)
//       parallel_for (i : [beg, end])
//         ...
//     }
//     __kitrt_nest_end(ctx)
//
// Here, beg and end are declared in the entry block of the function containing
// the loop. The call to __kitrt_nest_bounds is added to the lopo guard block.
//
void PrepareTapirNestParallel::updateLoopBounds(Loop &wrapperLoop, Loop &loop,
                                                Value *nestCtx) {
  auto sanityCheck = [](const Loop &wrapperLoop, const Loop &loop,
                        ScalarEvolution &se) {
    assert(wrapperLoop.getCanonicalInductionVariable() &&
           "Wrapper loop has canonical induction variable");

    assert(loop.getLoopPreheader() && "Loop has a preheader");
    assert(loop.getLoopLatch() && "Loop has a unique latch");
    assert(loop.getLatchCmpInst() &&
           "Loop has a unique latch compare instruction");
    assert(loop.getBounds(se).has_value() && "Loop has computable bounds");

    [[maybe_unused]] PHINode *iv = loop.getCanonicalInductionVariable();
    assert(iv && "Loop has canonical induction variable");

    [[maybe_unused]] BasicBlock *guard =
        getLoopGuardBlock(loop, /*allowDeadEndExits=*/true);
    assert(guard && "Loop has a guard block");
    assert(isa<BranchInst>(guard->getTerminator()) &&
           "Loop guard terminator is a branch instruction");
    assert(cast<BranchInst>(guard->getTerminator())->isConditional() &&
           "Lopo guard terminator is a conditional branch instruction");

    [[maybe_unused]] Loop::LoopBounds bounds = *loop.getBounds(se);
    [[maybe_unused]] Instruction *inc = &bounds.getStepInst();
    assert(inc->getOperand(0) == iv &&
           "First operand of step instruction is indction variable");
    assert(isIntOne(inc->getOperand(1)) &&
           "Second operand of step instruction is 1");

    [[maybe_unused]] Value *finl = &bounds.getFinalIVValue();
    [[maybe_unused]] ICmpInst *cmp = loop.getLatchCmpInst();
    [[maybe_unused]] CmpInst::Predicate cmpOp = cmp->getPredicate();
    [[maybe_unused]] Value *cmp0 = cmp->getOperand(0);
    [[maybe_unused]] Value *cmp1 = cmp->getOperand(1);
    assert((cmpOp == CmpInst::ICMP_EQ || cmpOp == CmpInst::ICMP_NE) &&
           "Predicate of compare instruction is EQ");
    assert((cmp0 == inc || cmp1 == inc) &&
           "Operand of compare instruction is the step");
    assert((cmp0 == finl || cmp1 == finl) &&
           "Operand of compare instruction is the trip count");

    assert(getFunction(loop) && "The loop is in a function");
  };

  LLVM_DEBUG(
      dbgs()
      << "PrepareTapirNestParallel: Update inner loop induction variable\n");
  sanityCheck(wrapperLoop, loop, se);

  LLVMContext &ctx = getContext(loop);

  BasicBlock &guard = *getLoopGuardBlock(loop, /*allowDeadEndExits=*/true);
  BasicBlock &ph = *loop.getLoopPreheader();

  ICmpInst *cmp = loop.getLatchCmpInst();
  BranchInst *br = cast<BranchInst>(guard.getTerminator());

  PHINode *iv = loop.getCanonicalInductionVariable();
  Type *ivTy = iv->getType();
  Loop::LoopBounds bounds = *loop.getBounds(se);
  Value *oldStart = &bounds.getInitialIVValue();
  Value *oldEnd = &bounds.getFinalIVValue();

  // Since the loop has a canonical induction variable, has a guard, and the
  // latch compare instruction predicate is EQ, the final value of the induction
  // variable will be available before the loop guard condition is computed. We
  // can therefore, insert the new guard computation just before the terminator
  // of the guard.
  //
  // The allocas for the out parameters of the bound computation are also added
  // to the guard. Since the outer tapir loop will eventually be outlined, the
  // allocas will then be present in a function that will be executed on each
  // loop. The allocas will not be within any loop at that point. If we were to
  // add the allocas to the entry block of the function containing the loop, we
  // would be introducing a race.
  IRBuilder<> builder(guard.getTerminator());
  AllocaInst *slotB = builder.CreateAlloca(ivTy, /*array=*/nullptr, "prlll.s");
  AllocaInst *slotE = builder.CreateAlloca(ivTy, /*array=*/nullptr, "prlll.e");
  PHINode *tid = wrapperLoop.getCanonicalInductionVariable();
  Constant *level = toConstant(*getPerfectLevelAttr(loop) - 1, ctx);
  createCall(builder, KitFunc::kitrt_nest_bounds,
             {nestCtx, tid, oldStart, oldEnd, level, slotB, slotE}, "",
             /*allowArgCasts=*/true);

  LoadInst *newStart = builder.CreateLoad(ivTy, slotB);
  LoadInst *newEnd = builder.CreateLoad(ivTy, slotE);
  Value *newCmp = builder.CreateICmpULT(newStart, newEnd);

  br->setCondition(newCmp);
  iv->setIncomingValueForBlock(&ph, newStart);
  replaceMatchingOperands(*cmp, oldEnd, newEnd);

  se.forgetValue(iv);
  se.forgetValue(oldEnd);
  se.forgetLoop(&loop);

#ifndef NDEBUG
  dt.verify();
  se.verify();
#endif // NDEBUG
}

// Transform a tapir loop. Add the tapir.loop.prepared attribute to the loop and
// return true if anything other than this attribute was changed.
bool PrepareTapirNestParallel::run(TapirLoopInfo &tapirLoop) {
  auto sanityCheck = [](const TapirLoopInfo &tapirLoop) {
    const Loop &loop = *tapirLoop.getLoop();

    // These checks are mostly to ensure that the loop objects known to the
    // driver don't get corrupted when processing a function with many tapir
    // loops.
    assert(isTapirLoop(loop) && "Loop is a tapir loop");
    assert(!hasPreparedAttr(loop) && "Tapir loop has not been prepared");
    assert(hasPerfectDepthAttr(loop) && "Loop has perfect.depth attribute");
    assert(hasPerfectLevelAttr(loop) && "Loop has perfect.level attribute");
  };

  StringRef name = getName(*tapirLoop.getLoop());
  LLVM_DEBUG(dbgs() << "PrepareTapirNestParallel: BEGIN '" << name << "'\n");
  sanityCheck(tapirLoop);

  Loop *wrapperLoop = wrapWithTapirLoop(tapirLoop, dt, li, mssa, se, ti);
  Value *ctx = wrapLoopWithContext(*wrapperLoop);
  updateWrapperBounds(*wrapperLoop, ctx);

  // It may not be strictly necessary to process the loops from inside out, but
  // it may allow us to avoid recalculating TaskInfo after each loop is
  // serialized.
  SmallVector<Loop *, 4> loops = getTapirLoopsInNest(*wrapperLoop);
  for (unsigned i = loops.size() - 1; i > 0; --i) {
    Loop &loop = *loops[i];
    updateLoopBounds(*wrapperLoop, loop, ctx);
    serializeLoop(loop, dt);
  }

  // The depth of the outer loop will be 1 since all the inner loops will have
  // been serialized.
  addPerfectDepthAttr(*wrapperLoop, 1);
  addPreparedAttr(*wrapperLoop);

#ifndef NDEBUG
  dt.verify();
#endif // NDEBUG

  // FIXME: This is wasteful, but, at this time, there does not seem to be a
  // better way to do this.
  Function &f = *getFunction(*wrapperLoop);
  ti.recalculate(f, dt);

#ifndef NDEBUG
  // Check that the outermost loop after the tapir loop nest has been prepared
  // is a parallel loop. All inner loops must be serial loops.
  assert(getTaskIfTapirLoop(wrapperLoop, &ti) &&
         "Prepared wrapper loop is a tapir loop");
  for (unsigned i = 1, e = loops.size(); i < e; ++i)
    assert(!getTaskIfTapirLoop(loops[i], &ti) &&
           "Prepared loop is not a tapir loop");
#endif // NDEBUG

  LLVM_DEBUG(dbgs() << "PrepareTapirNestParallel: END '" << name << "'\n");

  return true;
}

//==------------------------------------------------------------------------==//

static bool prepareTapirNestSerial(Loop &loop, DominatorTree &dt,
                                   TaskInfo &ti) {
  // We will be lazy and just serialize everything here.
  //
  // ---------------------------------- FIXME ----------------------------------
  //
  // FIXME: This is really *NOT* what we ought to be doing. This gets rid of
  // tapir loops very early in the pipeline, but we may want to keep them around
  // to do some sort of analysis. Yes, these are loops with the serial tapir
  // target and are very unlikely to ever be seen in practice, but it is the
  // principle of the thing. But transforming this in some sort of sane way is
  // challenging.
  //
  // If we were to use the same approach that we use for parallel loops, we end
  // up with a loop nest context, calls to __kitrt_nest_threads and
  // __kitrt_nest_bounds even for serial loops. We know that this unnecessary
  // because __kitrt_nest_threads is guaranteed to return 1 and
  // __kitrt_nest_bounds will return the original bounds of the loop. This is a
  // lot of overhead for no good reasons. Besides, we want the output of the
  // serial tapir target to be no different than a serial compilation of the
  // code.
  //
  // Instead of wrapping the nest with a context object, we could simply wrap it
  // with a tapir loop that iterates over `[0, @llvm.kit.cpu.num.threads(1))`.
  // During early intrinsic lowering, the intrinsic call is replaced with 1,
  // following which the loop is discarded. The problem with this is that the
  // loop is discarded by a combination of instcombine and simplifycfg, neither
  // of which know about tapir loops. When they remove the backedge etc, we are
  // left with a detach-reattach pair that are not part of a loop - something
  // that we do not currently allow.
  //
  // A third option would be to tile the innermost tapir loop in the nest with
  // the outer loop of the tile iterating over
  // `[0, @llvm.kit.cpu.num.threads(1))`. This would work a bit better, but
  // again, it is a lot of work to eventually get back to a standard serial loop
  // nest.
  //
  // We could try only serializing the innermost loop in the nest, which is an
  // awkward compromise at best. Besides, for tapir loops of depth 1, we would
  // anyway just be getting rid of the tapir loop anyway.
  //
  // Given what an annoying mess this has become, we just serialize everything
  // for now, with the intention of coming back to this at some point. It does
  // render the serial tapir target redundant since, by the time we get to
  // loop-spawning, there will be no tapir loops left to spawn. That is
  // unfortunate, but we will stick with that for now.
  // ---------------------------------------------------------------------------

  assert(isTopLevelTapirLoop(loop) && "Loop is a top-level tapir loop");

  SmallVector<Loop *, 4> loops = getTapirLoopsInNest(loop);
  for (unsigned i = loops.size(); i > 0; --i) {
    // We don't pass TaskInfo because recalculating it currently involves
    // recalculating it for the entire function. If we have to incur that cost,
    // it might as well be once per loop nest instead of once per loop.
    serializeLoop(*loops[i - 1], dt);
  }

  // FIXME: There is wasteful, but there is currently no better way to update
  // TaskInfo.
  Function &f = *getFunction(loop);
  ti.recalculate(f, dt);

#ifndef NDEBUG
  ti.verify(dt);
#endif // NDEBUG

  return true;
}

bool llvm::detail::prepareTapirNestCPU(TapirLoopInfo &tapirLoop,
                                       DominatorTree &dt, LoopInfo &li,
                                       MemorySSA &mssa, ScalarEvolution &se,
                                       TaskInfo &ti) {
  Loop &loop = *tapirLoop.getLoop();
  if (*getTargetAttr(loop) == TTID::Serial)
    return prepareTapirNestSerial(loop, dt, ti);
  else
    return PrepareTapirNestParallel(dt, li, mssa, se, ti).run(tapirLoop);
}
