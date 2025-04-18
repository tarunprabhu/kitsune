//===- ModuleSummaryAnalysis.cpp - Module summary index builder -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An interface for information about the tapir targets needed by a module.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TapirTargetAnalysis.h"

namespace llvm {

AnalysisKey TapirTargetAnalysis::Key;

TapirTargetAnalysis::TapirTargetAnalysis(std::optional<TapirTargetOptions> o) {
  if (o)
    ttOpts = o->clone();
}

TapirTargetInfo TapirTargetAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  return TapirTargetInfo(ttOpts.get());
}

} // namespace llvm
