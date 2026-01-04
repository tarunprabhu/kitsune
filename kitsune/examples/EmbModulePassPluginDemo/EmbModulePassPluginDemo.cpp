//===- EmbBitcodePassPluginDemo.cpp - Pass plugin for embedded bitcode ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple pass plugin to demonstrate the use of a pass plugin containing an
// embedded bitcode pass.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/EmbModulePass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

// Pass that prints the names of all functions in an embedded bitcode module.
struct EmbFuncNamesPass : public EmbModulePass<EmbFuncNamesPass> {
  bool run(TTID tt, Module &m, Module &hostM, ModuleAnalysisManager &hostMAM) {
    for (Function &f : m.functions())
      outs() << f.getName() << "\n";
    return false;
  }

  using EmbModulePass<EmbFuncNamesPass>::run;
};

template <typename Pass>
static void registerPass(ModulePassManager &pm, OptimizationLevel) {
  pm.addPass(Pass());
}

template <typename Pass>
static bool registerPassInPipeline(ModulePassManager &pm) {
  pm.addPass(Pass());
  return true;
}

static bool parsePassPipeline(StringRef name, ModulePassManager &pm,
                              ArrayRef<PassBuilder::PipelineElement>) {
  if (name == "emb-func-names")
    return registerPassInPipeline<EmbFuncNamesPass>(pm);
  return false;
}

static void registerPasses(PassBuilder &pb) {
  pb.registerKitsunePostTapirEarlyEPCallback(registerPass<EmbFuncNamesPass>);
  pb.registerPipelineParsingCallback(parsePassPipeline);
}

extern "C" LLVM_ATTRIBUTE_WEAK ::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "KitEmbBitcodePassPluginDemo",
          LLVM_VERSION_STRING, registerPasses};
}
