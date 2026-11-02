//===- PrepareTapirLoops.cpp - Prepare tapir loops for lowering -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform tapir loop nests to a form suitable for parallel execution. Note
// that this pass intentionally operates on a nests rooted at a top-level tapir
// loop. A top-level tapir loop for which one of the following hold:
//
//   - The loop does not have any loops as ancestors
//   - None of the ancestors of the loop are tapir loops
//
// This will only transform loop nests of tapir-depth <= 3. Here we have used
// "tapir-depth" to mean the number of nested tapir loops. The nest itself can
// have a depth greater than 3, but no more than three of those loops can be
// tapir loops. The usual constraints on the parents of tapir loops hold. For
// instance, the following are permitted:
//
//     for (...)           forall (...)        forall (...)
//       forall (...)        forall (...)        forall (...)
//         forall (...)        forall (...)        forall (...)
//           ...                 ...                 for (...)
//                                                     ...
//
// However, the following are not:
//
//     forall (...)        forall (...)
//       forall (...)        for (...)
//         forall (...)        forall (...)
//           forall (...)        ...
//             ...
//
// The specific transformations that are performed are described in the files
// where the transformation is implemented.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/PrepareTapirLoops.h"
#include "LoopWrapping.h"
#include "PrepareTapirLoopsImpl.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/Reductions.h"
#include "kitsune/Core/TTUtils.h"
#include "kitsune/Core/ValueUtils.h"
#include "kitsune/Passes/PassUtils.h"
#include "kitsune/Shared/RTNestUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#define DEBUG_TYPE "kit-prepare"

using namespace llvm;

// Perform sanity checks on \p tapirLoop that are to be performed on both
// reduction, and non-reduction, tapir loops.
static bool checkCommon(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                        LoopInfo &li, ScalarEvolution &se) {
  const Loop &root = *tapirLoop.getLoop();

  if (!root.isLoopSimplifyForm())
    return emitDiagnostic(root, DiagID::ErrLoopNotSimplifyForm);

  std::optional<unsigned> depth = getPerfectDepthAttr(root);
  if (!depth.has_value())
    return emitDiagnostic(root, DiagID::ErrAttrMissing,
                          getAttrName(LoopAttrKind::PerfectDepth));
  else if (*depth < 1 || *depth > NEST_MAX_DEPTH)
    return emitDiagnostic(root, DiagID::ErrTapirLoopNestInvalidDepth, 1,
                          NEST_MAX_DEPTH);

  SmallVector<const Loop *, 4> loops = getTapirLoopsInNest(root);
  for (const Loop *loop : loops) {
    if (!loop->getCanonicalInductionVariable())
      return emitDiagnostic(*loop, DiagID::ErrTapirLoopIVNotCanonical);

    std::optional<Loop::LoopBounds> bounds = loop->getBounds(se);
    if (!bounds.has_value())
      return emitDiagnostic(*loop, DiagID::ErrTapirLoopUncomputableBounds);
    else if (!loop->isLoopInvariant(&bounds->getFinalIVValue()))
      return emitDiagnostic(*loop, DiagID::ErrTapirLoopNoFiniteTripCount);

    if (!hasPerfectLevelAttr(*loop))
      return emitDiagnostic(*loop, DiagID::ErrAttrMissing,
                            getAttrName(LoopAttrKind::PerfectLevel));

    if (!getLoopGuardBlock(*loop, /*allowDeadEndExits=*/true))
      return emitDiagnostic(*loop, DiagID::ErrTapirLoopNoGuardBlock);

    if (!getLoopBypassBlock(*loop, /*allowDeadEndExits=*/true))
      return emitDiagnostic(*loop, DiagID::ErrTapirLoopNoBypassBlock);
  }

  return true;
}

// Check a non-reduction tapir loop.
static bool checkParallelLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                              LoopInfo &li, ScalarEvolution &se) {
  if (!checkCommon(tapirLoop, dt, li, se))
    return false;

  if (!checkTapirLoopSafeToWrap(tapirLoop, dt, li, se))
    return false;

  return true;
}

static bool checkReductionLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                               LoopInfo &li, ScalarEvolution &se) {
  if (!checkCommon(tapirLoop, dt, li, se))
    return false;

  Loop &root = *tapirLoop.getLoop();
  SmallVector<ReductionInfo, 1> redxns = collectReductions(root);

  // The destination of a reduction loop must be loop invariant. We currently
  // do not support loop-variant destinations.
  for (const ReductionInfo &redxn : redxns) {
    Value *dest = redxn.getDest();
    if (!root.isLoopInvariant(dest))
      return emitDiagnostic(root, DiagID::ErrReduceDestLoopVariant, dest);
  }

  // The transformations for parallel reductions, especially on the GPU, might
  // result in incorrect code if the destination of the reduction is used
  // anywhere except in a reduction intrinsic. This is certainly the case for
  // side-effecting uses such as passing it to printf. But it is less clear if
  // this is true in other cases. To be safe, we require the only uses of the
  // destination of a reduction to be in an equivalent reduction intrinsic. We
  // cannot require exactly one use because the reduction intrinsic could be in
  // an unrolled sequential loop within this tapir loop, which would appear as
  // multiple uses, though that case is not actually an error.
  SmallDenseMap<Value *, SmallVector<Instruction *, 1>> usesOfDest;
  for (const ReductionInfo &redxn : redxns)
    for (const Use &use : redxn.getDest()->uses())
      if (auto *inst = dyn_cast<Instruction>(use.getUser()))
        if (isInLoop(*inst, root, li))
          usesOfDest[redxn.getDest()].push_back(inst);

  for (const auto &[dest, uses] : usesOfDest) {
    if (uses.size() == 1)
      continue;

    for (Instruction *inst : uses) {
      auto *call = dyn_cast<CallInst>(inst);
      if (!call || call->getIntrinsicID() != Intrinsic::kit_reduce)
        return emitDiagnostic(root, DiagID::ErrReduceDestUsedInLoop,
                              getName(*dest));

      // At this point, the instruction is a reduce intrinsic.
      const ReductionInfo redxn(call);
      if (redxn.getValue() == dest || redxn.getUnit() == dest ||
          redxn.getReducer() == dest)
        return emitDiagnostic(root, DiagID::ErrReduceDestUsedInLoop,
                              getName(*dest));

      SmallVector<Value *, 0> extraArgs = redxn.getExtraArgs();
      for (Value *arg : extraArgs)
        if (arg == dest)
          return emitDiagnostic(root, DiagID::ErrReduceDestUsedInLoop,
                                getName(*dest));
    }

    // If the destination of a reduction is used in two different reductions in
    // the loop, the two reductions must have the same operator.
    SmallSet<ReduceOp, 1> reduceOpsForDest;
    for (Instruction *use : uses)
      if (auto *call = dyn_cast<CallInst>(use))
        if (call->getIntrinsicID() == Intrinsic::kit_reduce)
          reduceOpsForDest.insert(ReductionInfo(call).reduceOp);
    if (reduceOpsForDest.size() > 1)
      return emitDiagnostic(root, DiagID::ErrReduceDestMultipleOps,
                            getName(*dest));
  }

  if (isGPUTT(*getTargetAttr(root))) {
    // We only need the top-level loop in the nest to have a non-dead-end exit
    // block because the lowering will involve allocating temporary GPU memory
    // which must be freed in the loop exit block (it is allocated in the
    // preheader, but that is guaranteed to exist because we require the loop to
    // be in simplify form).
    if (!getUniqueNonDeadEndExitBlock(root))
      return emitDiagnostic(root,
                            DiagID::ErrTapirLoopNoUniqueNonDeadEndExitBlock);
  }

  // We don't currently support reductions where the value being reduced is
  // passed by pointer. This will most likely happen with custom reductions. On
  // CPU, this may be ok, but getting this right on GPUs is trickier, so we just
  // don't allow it anywhere.
  for (const ReductionInfo &redxn : redxns)
    if (redxn.getType()->isPointerTy())
      return emitDiagnostic(root, DiagID::ErrNYI,
                            "Reductions with values passed by pointer");

  return true;
}

static bool check(TapirLoopInfo &tapirLoop, DominatorTree &dt, LoopInfo &li,
                  ScalarEvolution &se) {
  Loop &loop = *tapirLoop.getLoop();
  if (hasReductionAttr(loop))
    return checkReductionLoop(tapirLoop, dt, li, se);
  else
    return checkParallelLoop(tapirLoop, dt, li, se);
}

static bool prepare(TapirLoopInfo &tapirLoop, DominatorTree &dt, LoopInfo &li,
                    MemorySSA &mssa, ScalarEvolution &se, TaskInfo &ti) {
  Loop &root = *tapirLoop.getLoop();
  TTID tt = *getTargetAttr(root);

  bool changed = false;
  if (isCPUTT(tt))
    changed |= detail::prepareTapirNestCPU(tapirLoop, dt, li, mssa, se, ti);
  else if (isGPUTT(tt))
    changed |= detail::prepareTapirNestGPU(root);
  else if (tt != TTID::Custom)
    llvm_unreachable("Unknown tapir target category");

  return changed;
}

static bool prepare(Loop &loop, DominatorTree &dt, LoopInfo &li,
                    MemorySSA &mssa, OptimizationRemarkEmitter &ore,
                    ScalarEvolution &se, TaskInfo &ti) {
  LLVM_DEBUG(dbgs() << "PrepareTapirLoop: Preparing loop '" << getName(loop)
                    << "'\n");

  // Even if the loop is recognized as a tapir loop, if it does not have the
  // correct structure, the transformation that must be performed by this pass
  // will be difficult, if not impossible to perform. Therefore, check this
  // early, and fail immediately. See the comment above the call to check() for
  // a discussion on why we choose to fail instead of producing working, even if
  // slow, code in such cases.
  Task *task = getTaskIfTapirLoopStructure(&loop, &ti);
  if (!task) {
    emitDiagnostic(loop, DiagID::ErrTapirLoopNoTask);
    exitOnError();
  }

  PredicatedScalarEvolution pse(se, loop);
  TapirLoopInfo tapirLoop(&loop, task);

  // Setup the tapir loop object. These must be done before we check if the
  // tapir loop can be transformed, otherwise, the check will definitely fail
  // with spurious errors. We do this early to separate the tasks of setting up
  // the object and checking the loop rather than having the two be
  // interspersed.
  tapirLoop.collectIVs(pse, DEBUG_TYPE, &ore);
  tapirLoop.getOrCreateTripCount(pse, DEBUG_TYPE, &ore);

  // If the tapir loop is such that it cannot be transformed for parallel
  // execution, the entire compilation should fail. At the time of writing this,
  // Kitsune is very much a research prototype, not a production-quality
  // compiler (or even remotely close to it). The goal is not to always produce
  // code that runs, but to push the envelope on the kinds of optimizations that
  // can be performed. Given this objective, it makes more sense to fail if a
  // transformation could not be performed, rather than produce working, albeit
  // slow, code.
  if (!check(tapirLoop, dt, li, se))
    exitOnError();

  return prepare(tapirLoop, dt, li, mssa, se, ti);
}

PreservedAnalyses PrepareTapirLoopsPass::run(Function &f,
                                             FunctionAnalysisManager &am) {
  DominatorTree &dt = am.getResult<DominatorTreeAnalysis>(f);
  LoopInfo &li = am.getResult<LoopAnalysis>(f);
  MemorySSA &mssa = am.getResult<MemorySSAAnalysis>(f).getMSSA();
  OptimizationRemarkEmitter &ore =
      am.getResult<OptimizationRemarkEmitterAnalysis>(f);
  ScalarEvolution &se = am.getResult<ScalarEvolutionAnalysis>(f);
  TaskInfo &ti = am.getResult<TaskAnalysis>(f);

  bool changed = false;
  for (Loop *loop : getTapirLoops(li))
    if (isTopLevelTapirLoop(*loop) && !hasPreparedAttr(*loop))
      changed |= prepare(*loop, dt, li, mssa, ore, se, ti);

  if (!changed)
    return getPreservedAnalysesAll();
  return getPreservedAnalysesLoop(/*preservedMSSA=*/true);
}
