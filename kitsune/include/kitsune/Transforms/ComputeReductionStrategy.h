//===- ComputeReductionStrategy.h - Compute reduction strategy -*- C++ -*--===//
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

#ifndef KITSUNE_TRANSFORMS_COMPUTE_REDUCTION_STRATEGY_H
#define KITSUNE_TRANSFORMS_COMPUTE_REDUCTION_STRATEGY_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Compute the strategy to use for reductions.
class ComputeReductionStrategyPass
    : public PassInfoMixin<ComputeReductionStrategyPass> {
public:
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &am);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_COMPUTE_REDUCTION_STRATEGY_H
