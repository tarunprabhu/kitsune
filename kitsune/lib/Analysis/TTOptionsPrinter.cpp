//==- TTOptionsPrinter.cpp - Pass that prints the TTOptions object ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass that prints the TTOptions object.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Analysis/TTOptionsPrinter.h"
#include "kitsune/Core/TTOptions.h"

using namespace llvm;

#define DEBUG_TYPE "kit-print-tt-options"

PreservedAnalyses TTOptionsPrinterPass::run(Module &m,
                                            ModuleAnalysisManager &am) {
  tto.print(outs(), /*all=*/true);
  return PreservedAnalyses::all();
}
