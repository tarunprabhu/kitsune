//===- GuardLoop.h - Ensure that a loop is guarded --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Ensure that a loop is guarded.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_GUARD_LOOP_H
#define KITSUNE_TRANSFORMS_GUARD_LOOP_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class LPMUpdater;

/// \ingroup kitsune
/// Add a guard around a tapir loop if it is not already guarded.
class GuardLoopPass : public PassInfoMixin<GuardLoopPass> {
public:
  PreservedAnalyses run(Loop &loop, LoopAnalysisManager &am,
                        LoopStandardAnalysisResults &ar, LPMUpdater &updater);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_GUARD_LOOP_H
