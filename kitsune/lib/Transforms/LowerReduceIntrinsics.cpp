//===- LowerReduceIntrinsics.cpp - Lower Kitsune's reduce intrinsics ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune's reduce intrinsics.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/LowerReduceIntrinsics.h"
#include "LowerReduceIntrinsicsCore.h"

#define DEBUG_TYPE "kit-lower-reduce-intrinsics"

using namespace llvm;

PreservedAnalyses LowerReduceIntrinsicsPass::run(Module &m,
                                                 ModuleAnalysisManager &am) {
  if (detail::lowerReduceIntrinsicsCore(m))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
