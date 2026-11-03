//===- LocalizeReductions.cpp - Compute local reductions --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compute per-thread reductions in prepared tapir loops. At the end of the
// thread, accumulate this contribution to the original destination.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_LOCALIZE_REDUCTIONS_H
#define KITSUNE_TRANSFORMS_LOCALIZE_REDUCTIONS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Compute per-thread reductions in prepared tapir loops. At the end of the
/// thread, accumulate this contribution to the original destination.
class LocalizeReductionsPass : public PassInfoMixin<LocalizeReductionsPass> {
public:
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &am);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_LOCALIZE_REDUCTIONS_H
