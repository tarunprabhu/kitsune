//===- PassUtils.cpp - Utilities for LLVM passes --------------------------===//
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

#include "kitsune/Passes/PassUtils.h"

using namespace llvm;

PreservedAnalyses llvm::getPreservedAnalysesAll() {
  return PreservedAnalyses::all();
}

PreservedAnalyses llvm::getPreservedAnalysesCFG() {
  PreservedAnalyses pa;
  pa.preserveSet<CFGAnalyses>();
  return pa;
}

PreservedAnalyses llvm::getPreservedAnalysesNone() {
  return PreservedAnalyses::none();
}
