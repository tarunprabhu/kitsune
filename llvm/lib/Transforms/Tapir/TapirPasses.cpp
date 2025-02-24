//===- TapirPasses.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is purely temporary. The contents were moved from Tapir.cpp because
// that file was refactored to be part of libLLVMTapirCommon.a. The contents of
// this are only the pass initializations which is only relevant for the old
// pass manager which should probably be removed anyway.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Transforms/Tapir.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Transforms/Tapir.h"

using namespace llvm;

/// initializeTapirOpts - Initialize all passes linked into the
/// TapirOpts library.
void llvm::initializeTapirOpts(PassRegistry &Registry) {
  initializeLoopSpawningTIPass(Registry);
  initializeLowerTapirToTargetPass(Registry);
  initializeTaskCanonicalizePass(Registry);
  initializeTaskSimplifyPass(Registry);
  initializeDRFScopedNoAliasWrapperPassPass(Registry);
  initializeLoopStripMinePass(Registry);
  initializeSerializeSmallTasksPass(Registry);
}
