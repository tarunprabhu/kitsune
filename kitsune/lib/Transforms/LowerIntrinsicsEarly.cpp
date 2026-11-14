//===- LowerIntrinsicsEarly.cpp - Lower Kitsune's intrinsics --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower Kitsune's intrinsics in the middle-end. This only lowers a subset of
// intrinsics that must be lowered before codegen.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/LowerIntrinsicsEarly.h"
#include "LowerIntrinsics.h"

#define DEBUG_TYPE "kit-lower-intrinsics-early"

using namespace llvm;

PreservedAnalyses LowerIntrinsicsEarlyPass::run(Function &f,
                                                FunctionAnalysisManager &am) {
  detail::IntrinsicKinds kindsToLower =
      detail::IntrinsicKind::SerialNumThreads |
      detail::IntrinsicKind::CPUReduce;
  if (detail::LowerIntrinsicsImpl(tto, kindsToLower).runOn(f))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
