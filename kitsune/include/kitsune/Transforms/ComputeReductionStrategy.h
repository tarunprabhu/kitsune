//===- ComputeReductionStrategy.h - Compute reduction strategy -*- C++ -*--===//
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

#ifndef KITSUNE_TRANSFORMS_COMPUTE_REDUCTION_STRATEGY_H
#define KITSUNE_TRANSFORMS_COMPUTE_REDUCTION_STRATEGY_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class LPMUpdater;

/// \ingroup kitsune
/// Compute a suitable strategy for tapir loops that perform reductions. If
/// calls to Kitsune's reduce intrinsics in the loop have a strategy of "auto",
/// these will be replaced with the computed strategy.
class ComputeReductionStrategyPass
    : public PassInfoMixin<ComputeReductionStrategyPass> {
public:
  PreservedAnalyses run(Loop &loop, LoopAnalysisManager &am,
                        LoopStandardAnalysisResults &ar, LPMUpdater &updater);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_COMPUTE_REDUCTION_STRATEGY_H
