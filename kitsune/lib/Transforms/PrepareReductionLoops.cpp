//===- PrepareReductionLoops.cpp - Transform tapir reduction loops --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform tapir loops that perform reductions to a form that is suitable for
// parallel execution. The actual transformations are carried out in different
// implementation files specialized for CPU and GPU. This defines utilities that
// are shared by those specializations.
//
//===----------------------------------------------------------------------===//

#include "PrepareReductionLoops.h"
#include "LoopWrapping.h"
#include "LowerReduceIntrinsics.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/IRBuilderUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/TTUtils.h"
#include "kitsune/Core/ValueUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

#define DEBUG_TYPE "kit-prepare"

using namespace llvm;

template <typename... Args>
static bool complain(const Loop &loop, DiagID diag, Args &&...args) {
  emitDiagnostic(loop, diag, args...);
  return false;
}

static bool lowerReduce0Intrs(const SmallVectorImpl<ReductionInfo> &redxns) {
  bool changed = false;
  for (const ReductionInfo &redxn : redxns)
    changed |= detail::lowerReduce0Intr(redxn.call);
  return changed;
}

static bool
prepareReductionLoopSerial(Loop &loop,
                           const SmallVectorImpl<ReductionInfo> &redxns) {
  bool changed = lowerReduce0Intrs(redxns);
  addPreparedAttr(loop);
  return changed;
}

static bool
prepareReductionLoopCPU(TapirLoopInfo &tapirLoop,
                        const SmallVectorImpl<ReductionInfo> &redxns,
                        DominatorTree &dt, LoopInfo &li, MemorySSA &mssa,
                        ScalarEvolution &se, TaskInfo &ti) {
  bool changed = false;
  changed |= detail::prepareReductionLoopForCPU(tapirLoop, redxns, dt, li, mssa,
                                                se, ti);
  changed |= lowerReduce0Intrs(redxns);
  return changed;
}

bool llvm::detail::prepareReductionLoop(TapirLoopInfo &tapirLoop,
                                        DominatorTree &dt, LoopInfo &li,
                                        MemorySSA &mssa, ScalarEvolution &se,
                                        TaskInfo &ti) {
  auto sanityCheck = [](const Loop &loop) -> void {
    // These checks are mostly to ensure that the loop objects known to the
    // driver don't get corrupted when processing a function with many tapir
    // loops.
    assert(isTapirLoop(loop) &&
           "Loop with reduction attribute is a tapir loop");
    assert(hasReductionAttr(loop) &&
           "Loop is expected to have the reduction attribute");
    assert(!hasPreparedAttr(loop) &&
           "Tapir reduction loop has not been prepared");
  };

  Loop &loop = *tapirLoop.getLoop();
  sanityCheck(loop);

  // If the loop does not perform any actual reductions, mark it as prepared,
  // but return false because only the metadata will have changed.
  SmallVector<ReductionInfo, 1> redxns = collectReductions(loop);
  if (redxns.empty()) {
    addPreparedAttr(loop);
    return false;
  }

  TTID tt = *getTargetAttr(loop);
  if (tt == TTID::Serial)
    return prepareReductionLoopSerial(loop, redxns);
  else if (isCPUTT(tt))
    return prepareReductionLoopCPU(tapirLoop, redxns, dt, li, mssa, se, ti);
  else if (isGPUTT(tt))
    return detail::prepareReductionLoopForGPU(loop, redxns);
  else
    llvm_unreachable("prepareReductionLoop: TT is neither CPU nor GPU-centric");
}

bool llvm::detail::checkReductionLoop(TapirLoopInfo &tapirLoop,
                                      DominatorTree &dt, LoopInfo &li) {
  if (!checkTapirLoopSafeToWrap(tapirLoop, dt, li))
    return false;

  // If this is not a top-level tapir loop, it is nested within another tapir
  // loop.
  // FIXME: Add support for reductions in nested tapir loops.
  Loop &loop = *tapirLoop.getLoop();
  if (!isTopLevelTapirLoop(loop))
    return complain(loop, DiagID::ErrGeneric,
                    "tapir reduction loop must be a top-level loop");

  SmallVector<ReductionInfo, 1> redxns = collectReductions(loop);

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
        if (isInLoop(*inst, loop, li))
          usesOfDest[redxn.getDest()].push_back(inst);

  for (const auto &[dest, uses] : usesOfDest) {
    if (uses.size() == 1)
      continue;

    std::optional<ReduceOp> reduceOp;
    for (Instruction *inst : uses) {
      auto *call = dyn_cast<CallInst>(inst);
      if (!call || call->getIntrinsicID() != Intrinsic::kit_reduce_0)
        return complain(loop, DiagID::ErrReduceDestUsedInLoop, getName(*dest));

      // At this point, the instruction is a reduce intrinsic.
      const ReductionInfo redxn(call);
      if (redxn.getValue() == dest || redxn.getUnit() == dest ||
          redxn.getReducer() == dest)
        return complain(loop, DiagID::ErrReduceDestUsedInLoop, getName(*dest));

      SmallVector<Value *, 0> extraArgs = redxn.getExtraArgs();
      for (Value *arg : extraArgs)
        if (arg == dest)
          return complain(loop, DiagID::ErrReduceDestUsedInLoop,
                          getName(*dest));

      ReduceOp op = redxn.reduceOp;
      if (reduceOp.has_value() && *reduceOp != op)
        return complain(loop, DiagID::ErrReduceDestMultipleOps, getName(*dest));

      reduceOp = op;
    }
  }

  // We don't currently support reductions where the value being reduced is
  // passed by pointer. This will most likely happen with custom reductions. On
  // CPU, this may be ok, but getting this right on GPUs is trickier, so we just
  // don't allow it anywhere.
  for (const ReductionInfo &redxn : redxns)
    if (redxn.getType()->isPointerTy())
      return complain(loop, DiagID::ErrNYI,
                      "Reductions with values passed by pointer");

  return true;
}
