//===- LowerKitReduceIntrinsics.cpp - Lower Kitsune's reduce intrinsics ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune's reduce intrinsics.
//
// Unlike most other intrinsics, this is done as part of the middle end. The
// reduction may involve generation of LLVM loops, or even tapir loops.
// Typically, several optimization passes will have to be run after this pass
// to ensure that any code that is generated here is optimized.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/LowerKitReduceIntrinsics.h"
#include "LowerKitReduceIntrinsicsCore.h"

#define DEBUG_TYPE "kit-lower-reduce-intrinsics"

using namespace llvm;

PreservedAnalyses LowerKitReduceIntrinsicsPass::run(Module &m,
                                                    ModuleAnalysisManager &am) {
  if (detail::lowerKitReduceIntrinsicsCore(m))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
