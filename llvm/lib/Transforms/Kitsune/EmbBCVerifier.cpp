//=- EmbBCVerifier.h - Verify embedded bitcode ------------------------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utiltiies and passes to verify embedded bitcode.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Kitsune/EmbBCVerifier.h"
#include "llvm/Analysis/TapirTargetAnalysis.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/KitsuneUtils.h"

#define DEBUG_TYPE "verify-emb-bc"

using namespace llvm;

bool llvm::verifyEmbBC(TapirTargetID tt, Module &hostM, raw_ostream *os) {
  if (std::unique_ptr<Module> embM = getEmbeddedModule(tt, hostM))
    return verifyModule(*embM, &errs());
  return false;
}

bool llvm::verifyAllEmbBC(Module &hostM, raw_ostream *os) {
  EmbeddedModulesMapTy embMs = getEmbeddedModules(hostM);
  for (const auto &[tt, embM] : embMs)
    if (verifyModule(*embM, &errs()))
      return true;
  return false;
}

static constexpr StringRef errMsg =
    "Broken embedded bitcode module found, compilation aborted!";

static bool verifyAllEmbBC(Module &hostM, const TapirTargetInfo &tgi) {
  if (tgi.hasID() and llvm::verifyAllEmbBC(hostM, &errs()))
    report_fatal_error(errMsg);
  return false;
}

PreservedAnalyses VerifyAllEmbBCPass::run(Module &hostM,
                                          ModuleAnalysisManager &mam) {
  const TapirTargetInfo &tgi = mam.getResult<TapirTargetAnalysis>(hostM);

  ::verifyAllEmbBC(hostM, tgi);

  return PreservedAnalyses::all();
}

bool VerifyEmbBCPass::run(TapirTargetID tt, Module &km, Module &hostM,
                          ModuleAnalysisManager &hostMAM) {
  if (not verifyModule(km, &errs()))
    report_fatal_error(errMsg);
  return false;
}

/// Legacy pass to compile the embedded bitcode to fat binaries.
class VerifyAllEmbBCLegacyPass : public ModulePass {
public:
  VerifyAllEmbBCLegacyPass() : ModulePass(ID) {
    initializeVerifyAllEmbBCLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "Verify embedded bitcode"; }

  void getAnalysisUsage(AnalysisUsage &au) const override {
    au.addRequired<TapirTargetAnalysisWrapperPass>();
  }

  bool runOnModule(Module &hostM) override {
    TapirTargetInfo tgi =
        getAnalysis<TapirTargetAnalysisWrapperPass>().getResult();

    verifyAllEmbBC(hostM, tgi);

    return false;
  }

public:
  static char ID;
};

char VerifyAllEmbBCLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(VerifyAllEmbBCLegacyPass, DEBUG_TYPE,
                      "Verify embedded bitcode", false, false)
INITIALIZE_PASS_DEPENDENCY(TapirTargetAnalysisWrapperPass)
INITIALIZE_PASS_END(VerifyAllEmbBCLegacyPass, DEBUG_TYPE,
                    "Verify embedded bitcode", false, false)

ModulePass *llvm::createVerifyAllEmbBCLegacyPass() {
  return new VerifyAllEmbBCLegacyPass();
}
