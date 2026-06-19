//===- PreLowerPrepare.h - Prepare tapir loops for lowering -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Miscellaneous collection of transformations to prepare tapir loops for
// lowering.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_PRE_LOWER_PREPARE_H
#define KITSUNE_TRANSFORMS_PRE_LOWER_PREPARE_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class LPMUpdater;

/// \ingroup kitsune
/// A miscellaneous collection of transformations to prepare tapir loops for
/// lowering.
class PreLowerPreparePass : public PassInfoMixin<PreLowerPreparePass> {
public:
  PreservedAnalyses run(Loop &loop, LoopAnalysisManager &am,
                        LoopStandardAnalysisResults &ar, LPMUpdater &updater);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_PRE_LOWER_PREPARE_H
