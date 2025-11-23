//===- ExtensionPointsTest.cpp - Tests for Kitsune's extension points  ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// FIXME:
//
// It would be better if this were tested as part of a pass plugin because that
// is what the extension points are actually useful for. If we were to keep with
// LLVM's directory layout, this pass plugin would have to be part of examples/
// that would be built conditionally. However, there are a number of things in
// kitsune/examples right now, most of which are probably outdated. I would
// rather not have a new example in there until we clean up the existing
// contents. For now, we just have a unit test that also checks that the
// extension points insert passes into the expected point in the pipeline. Once
// the examples directory is cleaned up, it may be better to have a regular
// LLVM lit test with a pass plugin rather than this unit test.

#include "kitsune/Core/Tapir.h"
#include "kitsune/Core/TargetUtils.h"
#include "kitsune/Frontend/KitsuneOptions.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

struct PreEarly : PassInfoMixin<PreEarly> {
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam) {
    return PreservedAnalyses::all();
  }
};

struct PreLate : PassInfoMixin<PreLate> {
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam) {
    return PreservedAnalyses::all();
  }
};

struct PostEarly : PassInfoMixin<PostEarly> {
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam) {
    return PreservedAnalyses::all();
  }
};

struct PostLate : PassInfoMixin<PostLate> {
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam) {
    return PreservedAnalyses::all();
  }
};

struct PostLast : PassInfoMixin<PostLast> {
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam) {
    return PreservedAnalyses::all();
  }
};

TEST(ExtensionPoints, all) {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  LLVMContext ctx;

  KitsuneOptions kitOpts;
  kitOpts.setTTID(TTID::Cuda);
  std::optional<TTOptions> tto =
      *TTOptions::create(kitOpts, OptznLevel::O2, FPOpFusionMode::Standard);

  // The analysis managers must be declared in this order so that they are
  // destroyed in the correct order due to inter-analysis-manager references
  LoopAnalysisManager lam;
  FunctionAnalysisManager fam;
  CGSCCAnalysisManager cgam;
  ModuleAnalysisManager mam;
  PassInstrumentationCallbacks pic;
  TargetMachine *tm = createHostTargetMachine(*tto);
  OptimizationLevel optLevel = OptimizationLevel::O2;

  PrintPassOptions printOpts;
  printOpts.Verbose = true;

  StandardInstrumentations si(ctx, false, false, printOpts);
  si.registerCallbacks(pic, &mam);

  PipelineTuningOptions pto;
  pto.TTOpts = tto;

  PassBuilder pb(tm, pto);
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  tm->registerPassBuilderCallbacks(pb);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  pb.registerKitsunePreTapirEarlyEPCallback(
      [](ModulePassManager &mpm, OptimizationLevel) {
        mpm.addPass(PreEarly());
      });
  pb.registerKitsunePreTapirLateEPCallback(
      [](ModulePassManager &mpm, OptimizationLevel) {
        mpm.addPass(PreLate());
      });
  pb.registerKitsunePostTapirEarlyEPCallback(
      [](ModulePassManager &mpm, OptimizationLevel) {
        mpm.addPass(PostEarly());
      });
  pb.registerKitsunePostTapirLateEPCallback(
      [](ModulePassManager &mpm, OptimizationLevel) {
        mpm.addPass(PostLate());
      });
  pb.registerKitsunePostTapirLastEPCallback(
      [](ModulePassManager &mpm, OptimizationLevel) {
        mpm.addPass(PostLast());
      });

  std::string buf;
  raw_string_ostream os(buf);

  ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(optLevel);
  mpm.printPipeline(os, [&pic](StringRef className) -> StringRef {
    StringRef passName = pic.getPassNameForClassName(className);
    return passName.empty() ? className : passName;
  });
  os << "\n";
  os.flush();

  // This is a rough approximation of how FileCheck would behave (see the note
  // at the top of this file).
  size_t preEarly = buf.find("PreEarly");
  size_t preLate = buf.find("PreLate");
  size_t loopSpawning = buf.find("LoopSpawning");
  size_t postEarly = buf.find("PostEarly");
  size_t prefetch = buf.find("Prefetching");
  size_t postLate = buf.find("PostLate");
  size_t generateCtors = buf.find("GenerateCtors");
  size_t postLast = buf.find("PostLast");

  EXPECT_TRUE(preEarly != std::string::npos);
  EXPECT_TRUE(preLate != std::string::npos && preLate > preEarly);
  EXPECT_TRUE(loopSpawning != std::string::npos && loopSpawning > preLate);
  EXPECT_TRUE(postEarly != std::string::npos && postEarly > loopSpawning);
  EXPECT_TRUE(prefetch != std::string::npos && prefetch > postEarly);
  EXPECT_TRUE(postLate != std::string::npos && postLate > prefetch);
  EXPECT_TRUE(generateCtors != std::string::npos && generateCtors > postLate);
  EXPECT_TRUE(postLast != std::string::npos && postLast > generateCtors);
}

} // namespace
