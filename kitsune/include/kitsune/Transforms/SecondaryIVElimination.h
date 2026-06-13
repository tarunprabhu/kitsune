//===- SecondaryIVElimination.h - Eliminate secondary indvars --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Eliminate non-primary induction variables from tapir loops. This ensures
// that all tapir loops have exactly one induction variable.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_SECONDARY_IV_ELIMINATION_H
#define KITSUNE_TRANSFORMS_SECONDARY_IV_ELIMINATION_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class LPMUpdater;

/// \ingroup kitsune
/// Eliminate non-primary induction variables from tapir loops.
class SecondaryIVEliminationPass
    : public PassInfoMixin<SecondaryIVEliminationPass> {
public:
  PreservedAnalyses run(Loop &loop, LoopAnalysisManager &am,
                        LoopStandardAnalysisResults &ar, LPMUpdater &updater);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_SECONDARY_IV_ELIMINATION_H
