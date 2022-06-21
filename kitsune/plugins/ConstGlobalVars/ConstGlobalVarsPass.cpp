/*
 * Copyright (c) 2022 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */

#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "ConstGlobalVarsContext.h"

using namespace llvm;

// This pass adds custom metadata to global variables that were declared
// const in the source.
class ConstGlobalVarsPass : public PassInfoMixin<ConstGlobalVarsPass> {
public:
  PreservedAnalyses run(Module& M, ModuleAnalysisManager&) {
    // The singleton object cannot be referenced earlier than this because
    // it may not have been created. It is only created once the
    // PluginASTAction is launched and its CreateASTConsumer() method called.
    // However, if no source files were parsed during the invocation, an
    // instance will not have been created.
    if (const auto *SharedCtxt = ConstGlobalVarsContext::getIfExists()) {
      LLVMContext &Ctxt = M.getContext();

      for (llvm::GlobalVariable &GV : M.globals()) {
        if (GV.hasName() and SharedCtxt->isConstGlobal(GV.getName())) {
          MDString *tag = MDString::get(Ctxt, "kitsune.const");
          GV.addMetadata("kitsune", *MDTuple::get(Ctxt, tag));
        }
      }
    }

    return PreservedAnalyses::all();
  }
};

// Register the pass as early as possible in the optimization pipeline.
static void registerPass(PassBuilder& PB) {
  PB.registerPipelineStartEPCallback([](ModulePassManager& MPM,
                                        PassBuilder::OptimizationLevel) {
    MPM.addPass(ConstGlobalVarsPass());
  });
}

extern "C" PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION,
          "ConstGlobalVarsPass",
          LLVM_VERSION_STRING,
          registerPass};
}
