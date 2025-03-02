//===- TaskCanonicalize.cpp - Tapir task simplification pass ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass canonicalizes Tapir tasks, in particular, to split blocks at
// taskframe.create intrinsics.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/TaskCanonicalize.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "task-canonicalize"

PreservedAnalyses TaskCanonicalizePass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  if (F.empty())
    return PreservedAnalyses::all();

  LLVM_DEBUG(dbgs() << "TaskCanonicalize running on function " << F.getName()
                    << "\n");

  bool Changed = splitTaskFrameCreateBlocks(F);
  if (!Changed)
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}
