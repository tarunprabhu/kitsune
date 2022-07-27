//==- AsyncPrefetch.h - Asynchronous prefetching for Tapir loops -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Adds asynchronous prefetch calls when appropriate before Tapir loops.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_ASYNCPREFETCHPASS_H
#define LLVM_TRANSFORMS_TAPIR_ASYNCPREFETCHPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// Add asynchronous prefetch calls Pass.
struct AsyncPrefetchPass : public PassInfoMixin<AsyncPrefetchPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

}

#endif
