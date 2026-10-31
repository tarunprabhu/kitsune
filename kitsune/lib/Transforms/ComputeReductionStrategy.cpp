//===- ComputeReductionStrategy.cpp - Compute reduction strategy ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Update the strategy in calls to Kitsune's reduce intrinsics when the strategy
// is "auto".
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/ComputeReductionStrategy.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/Reductions.h"
#include "kitsune/Core/TTUtils.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "kit-reduce-strategy"

using namespace llvm;

// The default strategy to use when reducing into the final result.
static ReductionStrategy defaultStrategyCPU = ReductionStrategy::ThreadLocal;
static ReductionStrategy defaultStrategyGPU = ReductionStrategy::GPUWarpShuffle;

static cl::opt<ReductionStrategy> clStrategyCPU(
    "kit-cpu-reduce-strategy",
    cl::desc("The strategy to use for tapir reduction loops on the CPU"),
    cl::init(defaultStrategyGPU), cl::value_desc("strategy"),
    cl::cat(cl::catKitClOpts),
    cl::values(clEnumValN(ReductionStrategy::Direct, "direct", ""),
               clEnumValN(ReductionStrategy::Naive, "naive", ""),
               clEnumValN(ReductionStrategy::ThreadLocal, "local", "")));

static cl::opt<ReductionStrategy> clStrategyGPU(
    "kit-gpu-reduce-strategy",
    cl::desc("The strategy to use for tapir reduction loops on the GPU"),
    cl::init(defaultStrategyCPU), cl::value_desc("strategy"),
    cl::cat(cl::catKitClOpts),
    cl::values(clEnumValN(ReductionStrategy::Direct, "direct", ""),
               clEnumValN(ReductionStrategy::Naive, "naive", ""),
               clEnumValN(ReductionStrategy::GPUSharedMemory, "shmem", ""),
               clEnumValN(ReductionStrategy::GPUWarpShuffle, "wshfl", ""),
               clEnumValN(ReductionStrategy::GPUWarpShuffleWithSharedMemory,
                          "wshflshmem", "")));

using Reductions = SmallVector<ReductionInfo, 1>;

static bool consistentRedxns(const Reductions &redxns) {
  auto match = std::adjacent_find(
      redxns.begin(), redxns.end(),
      [](const ReductionInfo &r1, const ReductionInfo &r2) -> bool {
        return r1.strategy != r2.strategy;
      });
  return match == redxns.end();
}

static bool isStrategyAuto(const Reductions &redxns) {
  assert(!redxns.empty() && "At least one reduction in loop");
  assert(consistentRedxns(redxns) && "Reduction strategies are consistent");

  return redxns[0].strategy == ReductionStrategy::Auto;
}

static ReductionStrategy computeStrategyCPU(const Loop &loop) {
  if (clStrategyCPU.getNumOccurrences())
    return clStrategyCPU;
  return defaultStrategyCPU;
}

static ReductionStrategy computeStrategyGPU(const Loop &loop) {
  if (clStrategyGPU.getNumOccurrences())
    return clStrategyGPU;
  return defaultStrategyGPU;
}

static ReductionStrategy computeStrategy(const Loop &loop) {
  assert(isTapirLoop(loop) && "Loop is a tapir loop");

  // TODO: We should consider performing an analysis on the loop to determine
  // a reduction strategy based on the computation being performed in the loop.
  // For now, we simply use the default strategy depending on the tapir target.
  TTID tt = *getTargetAttr(loop);
  if (isCPUTT(tt))
    return computeStrategyCPU(loop);
  else if (isGPUTT(tt))
    return computeStrategyGPU(loop);
  else
    llvm_unreachable("TTID not handled");
}

static bool updateReductionStrategy(Loop &loop, const Reductions &redxns) {
  if (redxns.empty() || !isStrategyAuto(redxns))
    return false;

  LLVMContext &ctx = getContext(loop);
  ReductionStrategy strategy = computeStrategy(loop);
  for (const ReductionInfo &redxn : redxns)
    redxn.call->setArgOperand(ReductionInfo::argNumStrategy,
                              toConstant(strategy, ctx));
  return true;
}

static bool check(const Loop &loop, const Reductions &redxns) {
  if (!consistentRedxns(redxns))
    return emitDiagnostic(loop, DiagID::ErrGeneric,
                          "Inconsistent reduction strategies");
  return true;
}

PreservedAnalyses
ComputeReductionStrategyPass::run(Loop &loop, LoopAnalysisManager &am,
                                  LoopStandardAnalysisResults &ar,
                                  LPMUpdater &updater) {
  if (isTapirLoop(loop) && hasReductionAttr(loop)) {
    Reductions redxns = collectReductions(loop);
    if (!redxns.empty() && check(loop, redxns))
      updateReductionStrategy(loop, redxns);
  }

  // Even if the any changes were made, they will only have updated an argument
  // in calls to Kitsune's reduce intrinsics. This will not invalidate any
  // analyses.
  return PreservedAnalyses::all();
}
