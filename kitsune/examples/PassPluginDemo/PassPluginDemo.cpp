//===- PassPluginDemo.cpp - Pass plugin for Kitsune's extension points ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple pass plugin to demonstrate the use of Kitsune-specific extension
// points. There is nothing here that is particularly different from LLVM's
// pass plugins except for the additional extension points that are available.
//
// We create a pass for every extension point specific to Kitsune. This is
// purely for demonstration. In a pass plugin, the user may create only as many
// passes as required. The same pass can also be registered with more than one
// extension point. For each pass, we also show how to register a pipeline
// parsing callback so the pass can be explicitly run by passing it to the
// -passes option in opt. See the tests in kitsune/test/plugins/pass-plugin to
// see how this is done (documentation may also be found in kitsune/docs).
//
// See kitsune/docs for more information about the organization of the kitsune
// and tapir pass pipelines and when, in those pipelines, the passes registered
// with the various extension points are run.
//
// The passes here are all module passes, but they could just as easily be
// function or loop passes.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

// Pass that is run early in the pre-tapir pipeline i.e. it will be run before
// the standard pre-tapir passes.
struct PreTapirEarlyPass : PassInfoMixin<PreTapirEarlyPass> {
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam) {
    return PreservedAnalyses::all();
  }
};

// Pass that is run late in the pre-tapir pipeline i.e. it will be run after the
// standard pre-tapir passes.
struct PreTapirLatePass : PassInfoMixin<PreTapirLatePass> {
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam) {
    return PreservedAnalyses::all();
  }
};

// Pass that is run early in the post-tapir pipeline i.e. it will be run before
// the standard post-tapir passes.
struct PostTapirEarlyPass : PassInfoMixin<PostTapirEarlyPass> {
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam) {
    return PreservedAnalyses::all();
  }
};

// Pass that is run late in the post-tapir pipeline i.e. it will be run after
// the standard post-tapir passes
struct PostTapirLatePass : PassInfoMixin<PostTapirLatePass> {
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam) {
    return PreservedAnalyses::all();
  }
};

// Pass that is run as late as possible in the post-tapir pipeline. This is
// guaranteed to run after all passes in the post-tapir pipeline.
struct PostTapirLastPass : PassInfoMixin<PostTapirLastPass> {
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam) {
    return PreservedAnalyses::all();
  }
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
  if (name == "pre-tapir-early")
    return registerPassInPipeline<PreTapirEarlyPass>(pm);
  else if (name == "pre-tapir-late")
    return registerPassInPipeline<PreTapirLatePass>(pm);
  else if (name == "post-tapir-early")
    return registerPassInPipeline<PostTapirEarlyPass>(pm);
  else if (name == "post-tapir-late")
    return registerPassInPipeline<PostTapirLatePass>(pm);
  else if (name == "post-tapir-last")
    return registerPassInPipeline<PostTapirLastPass>(pm);
  else
    return false;
}

static void registerPasses(PassBuilder &pb) {
  pb.registerKitsunePreTapirEarlyEPCallback(registerPass<PreTapirEarlyPass>);
  pb.registerKitsunePreTapirLateEPCallback(registerPass<PreTapirLatePass>);
  pb.registerKitsunePostTapirEarlyEPCallback(registerPass<PostTapirEarlyPass>);
  pb.registerKitsunePostTapirLateEPCallback(registerPass<PostTapirLatePass>);
  pb.registerKitsunePostTapirLastEPCallback(registerPass<PostTapirLastPass>);
  pb.registerPipelineParsingCallback(parsePassPipeline);
}

extern "C" LLVM_ATTRIBUTE_WEAK ::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "KitPassPluginDemo", LLVM_VERSION_STRING,
          registerPasses};
}
