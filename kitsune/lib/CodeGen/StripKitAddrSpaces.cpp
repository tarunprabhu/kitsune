//===- StripKitAddrSpaces.cpp - Strip Kitsune's address spaces ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Move pointers in Kitsune-specific address spaces to the default address
// space. This will mutate the types of the relevant entities in the module.
// This is done because the backends cannot currently handle Kitsune's address
// spaces.
//
//===----------------------------------------------------------------------===//

#include "kitsune/CodeGen/StripKitAddrSpaces.h"
#include "kitsune/Core/AddrSpace.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "kit-strip-addrspaces"

using namespace llvm;

namespace {

/// Pass, for the legacy pass manager, to strip Kitsune-specific address spaces
/// from pointers. This essentially puts the pointers into the default address
/// space.
class StripKitAddrSpacesLegacyPass : public ModulePass {
public:
  StripKitAddrSpacesLegacyPass() : ModulePass(ID) {
    initializeStripKitAddrSpacesLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Strip Kitsune address spaces";
  }

  void getAnalysisUsage(AnalysisUsage &au) const override {}

  bool runOnModule(Module &m) override { return stripKitAddrSpaces(m); }

public:
  static char ID;
};

} // namespace

char StripKitAddrSpacesLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(StripKitAddrSpacesLegacyPass, DEBUG_TYPE,
                      "Strip kitsune address spaces", false, false)
INITIALIZE_PASS_END(StripKitAddrSpacesLegacyPass, DEBUG_TYPE,
                    "Strip kitsune address spaces", false, false)

ModulePass *llvm::createStripKitAddrSpacesLegacyPass() {
  return new StripKitAddrSpacesLegacyPass();
}

PreservedAnalyses StripKitAddrSpacesPass::run(Module &m,
                                              ModuleAnalysisManager &mam) {
  if (stripKitAddrSpaces(m))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
