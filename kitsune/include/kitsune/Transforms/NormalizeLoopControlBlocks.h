//=- NormalizeLoopControlBlocks.h - Normalize loops pre-lowering --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Normalize the control blocks of tapir loops.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_NORMALIZE_LOOP_CONTROL_BLOCKS_H
#define KITSUNE_TRANSFORMS_NORMALIZE_LOOP_CONTROL_BLOCKS_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class LPMUpdater;

/// \ingroup kitsune
/// Normalize the control blocks of tapir loops prior to lowering.
class NormalizeLoopControlBlocksPass
    : public PassInfoMixin<NormalizeLoopControlBlocksPass> {
public:
  PreservedAnalyses run(Loop &loop, LoopAnalysisManager &am,
                        LoopStandardAnalysisResults &ar, LPMUpdater &updater);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_NORMALIZE_LOOP_CONTROL_BLOCKS_H
