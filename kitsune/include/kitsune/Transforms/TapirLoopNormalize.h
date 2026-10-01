//===- TapirLoopNormalize.h - Normalize tapir loops ------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Normalize tapir loops.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_TAPIR_LOOP_NORMALIZE_H
#define KITSUNE_TRANSFORMS_TAPIR_LOOP_NORMALIZE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Pass that normalizes a tapir loop.
class TapirLoopNormalizePass : public PassInfoMixin<TapirLoopNormalizePass> {
public:
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &am);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_TAPIR_LOOP_NORMALIZE_H
