//===- PassUtils.h - Utilities for LLVM passes ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM passes
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_PASSES_PASS_UTILS_H
#define KITSUNE_PASSES_PASS_UTILS_H

#include "kitsune/Passes/PassUtilsInternal.h"
#include "llvm/IR/Analysis.h"

namespace llvm {

/// Get a PreservedAnalyses object that indicates that all analyses are
/// preserved.
PreservedAnalyses getPreservedAnalysesAll();

/// Get a PreservedAnalyses object that indicates that all analyses except the
/// callgraph analyses have been preserved.
PreservedAnalyses getPreservedAnalysesCallGraph();

/// Get a PreservedAnalyses object that indicates that all analyses that depend
/// on the CFG are preserved.
PreservedAnalyses getPreservedAnalysesCFG();

/// Get a PreservedAnalyses object that indicates that all analyses have been
/// invalidated.
PreservedAnalyses getPreservedAnalysesNone();

} // namespace llvm

#endif // KITSUNE_PASSES_PASS_UTILS_H
