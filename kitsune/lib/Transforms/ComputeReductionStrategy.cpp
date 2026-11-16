//===- ComputeReductionStrategy.cpp - Compute reduction strategy ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compute the strategy to use for reductions.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/ComputeReductionStrategy.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/Reductions.h"
#include "kitsune/Core/TTUtils.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/Analysis/LoopInfo.h"

#define DEBUG_TYPE "kit-reduce-strategy"

using namespace llvm;

// The default strategy to use when reducing into the final result.
static ReductionStrategy defaultStrategyCPU = ReductionStrategy::Direct;
static ReductionStrategy defaultStrategyGPU = ReductionStrategy::GPUWarpShuffle;

static cl::opt<ReductionStrategy> clStrategyCPU(
    "kit-cpu-reduce-strategy",
    cl::desc("The strategy to use for tapir reduction loops on the CPU"),
    cl::init(defaultStrategyGPU), cl::value_desc("strategy"),
    cl::cat(cl::catKitClOpts),
    cl::values(clEnumValN(ReductionStrategy::Direct, "direct", "")));

static cl::opt<ReductionStrategy> clStrategyGPU(
    "kit-gpu-reduce-strategy",
    cl::desc("The strategy to use for tapir reduction loops on the GPU"),
    cl::init(defaultStrategyCPU), cl::value_desc("strategy"),
    cl::cat(cl::catKitClOpts),
    cl::values(clEnumValN(ReductionStrategy::Direct, "direct", ""),
               clEnumValN(ReductionStrategy::GPUSharedMemory, "shmem", ""),
               clEnumValN(ReductionStrategy::GPUWarpShuffle, "wshfl", ""),
               clEnumValN(ReductionStrategy::GPUWarpShuffleWithSharedMemory,
                          "wshflshmem", "")));

namespace {

using Reductions = SmallVector<ReductionInfo, 0>;

class ComputeStrategy {
protected:
  LoopInfo &li;

protected:
  ReductionStrategy computeStrategyCPU(const ReductionInfo &redxn,
                                       const Loop &loop);
  ReductionStrategy computeStrategyGPU(const ReductionInfo &redxn,
                                       const Loop &loop);
  ReductionStrategy computeStrategy(const ReductionInfo &redxn,
                                    const Loop &loop);
  bool updateStrategy(const ReductionInfo &redxn, ReductionStrategy strategy);
  Reductions getReductions(Loop &loop);

public:
  ComputeStrategy(LoopInfo &li) : li(li) {}

  bool runOn(Function &f);
};

} // namespace

ReductionStrategy
ComputeStrategy::computeStrategyCPU(const ReductionInfo &redxn,
                                    const Loop &loop) {
  // At this time, the only reduction strategy available on a CPU is "direct".
  // Even so, for consistency with the GPU implementation, we pretend as if we
  // actually have options.
  if (clStrategyCPU.getNumOccurrences())
    return clStrategyCPU;
  return defaultStrategyCPU;
}

ReductionStrategy
ComputeStrategy::computeStrategyGPU(const ReductionInfo &redxn,
                                    const Loop &loop) {
  // TODO: We should analyze the loop to calculate a suitable reduction strategy
  // to reduce the partial reductions computed by each warp/block into the final
  // result. For now, we just use the default, or the explicit strategy, if one
  // has been provided.
  if (clStrategyGPU.getNumOccurrences())
    return clStrategyGPU;
  return defaultStrategyGPU;
}

ReductionStrategy ComputeStrategy::computeStrategy(const ReductionInfo &redxn,
                                                   const Loop &loop) {
  TTID tt = redxn.tt;
  if (tt == TTID::Serial)
    return ReductionStrategy::Naive;
  else if (isCPUTT(tt))
    return computeStrategyCPU(redxn, loop);
  else if (isGPUTT(tt))
    return computeStrategyGPU(redxn, loop);
  else if (tt == TTID::Custom)
    return ReductionStrategy::Direct;
  llvm_unreachable("computeStrategy: Unknown TTID kind");
}

bool ComputeStrategy::updateStrategy(const ReductionInfo &redxn,
                                     ReductionStrategy strategy) {
  if (redxn.strategy != ReductionStrategy::Auto)
    return false;

  CallInst *call = redxn.call;
  LLVMContext &ctx = call->getContext();

  call->setArgOperand(ReductionInfo::argNumStrategy, toConstant(strategy, ctx));

  return true;
}

Reductions ComputeStrategy::getReductions(Loop &loop) {
  // The tapir loops in a nest will have been serialized, then wrapped with a
  // single tapir loop. The final reductions will been in this wrapper loop.
  // All other reductions will have been converted naive serial reductions. So
  // we only need to look for reductions in the outermost loop.
  auto collectFinalReductions = [](const Loop &loop) -> Reductions {
    Reductions redxns;
    for (BasicBlock *bb : getBlocksNotInSubLoops(loop))
      for (Instruction &inst : *bb)
        if (auto *call = dyn_cast<CallInst>(&inst))
          if (call->getIntrinsicID() == Intrinsic::kit_reduce)
            redxns.emplace_back(call);
    return redxns;
  };

  TTID tt = *getTargetAttr(loop);
  if (tt == TTID::Serial)
    return ::collectReductions(loop);
  else if (isCPUTT(tt))
    return collectFinalReductions(loop);
  else if (isGPUTT(tt))
    return collectFinalReductions(*getInnermostTapirLoopInNest(loop));
  else if (tt == TTID::Custom)
    return {};
  llvm_unreachable("ComputeStrategy::collectReductions: Unknown TT kind");
}

bool ComputeStrategy::runOn(Function &f) {
  bool changed = false;
  for (Loop *loop : li.getLoopsInPreorder())
    if (isTopLevelTapirLoop(*loop) && hasReductionAttr(*loop))
      for (const ReductionInfo &redxn : getReductions(*loop))
        changed |= updateStrategy(redxn, computeStrategy(redxn, *loop));
  return changed;
}

PreservedAnalyses
ComputeReductionStrategyPass::run(Function &f, FunctionAnalysisManager &am) {
  LoopInfo &li = am.getResult<LoopAnalysis>(f);

  ComputeStrategy(li).runOn(f);

  // At most, this will only have changed an argument in a call to Kitsune's
  // reduce intrinsic.
  return PreservedAnalyses::all();
}
