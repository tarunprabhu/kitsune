//===--- BlockingPrefetch.h - Block loop and issue prefetches ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Carries out an unconditional blocking prefetch transformation. This will
// block a Tapir loop and at iteration i, issue prefetches for data that will
// be used in iteration i + k.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_LOOPBLOCKINGPREFETCHPASS_H
#define LLVM_TRANSFORMS_TAPIR_LOOPBLOCKINGPREFETCHPASS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {

extern cl::opt<bool> EnableTapirLoopBlockingPrefetch;
extern cl::opt<unsigned> TapirBlockingPrefetchBlockCount;

/// The Blocking prefetch Pass.
class LoopBlockingPrefetchPass
    : public PassInfoMixin<LoopBlockingPrefetchPass> {
public:
  PreservedAnalyses run(Function &func, FunctionAnalysisManager& fam);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_TAPIR_LOOPBLOCKINGPREFETCHPASS_H
