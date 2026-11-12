//===- EmbOptimize.cpp - Optimize embedded modules ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Run a sequence of optimization passes on any embedded modules. Currently,
// this is just the standard sequence of optimization passes that are determined
// by the optimization level specified on the command-line. At some point, this
// may be replaced with a sequence that is specific to the tapir target that
// generated the embedded module.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/EmbOptimize.h"
#include "kitsune/Core/TTOptions.h"
#include "kitsune/Core/TTUtils.h"
#include "kitsune/Core/TargetUtils.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

#include "llvm/Support/IOSandbox.h"

#define DEBUG_TYPE "emb-optimize"

using namespace llvm;

static cl::opt<OptimizationLevel> clOptimizationLevel(
    cl::init(OptimizationLevel::O2), cl::Hidden,
    cl::values(
        clEnumValN(OptimizationLevel::O0, "emb-O0", "No optimizations"),
        clEnumValN(OptimizationLevel::O1, "emb-O1", "Some optimizations"),
        clEnumValN(OptimizationLevel::O2, "emb-O2", "Most optimizations"),
        clEnumValN(OptimizationLevel::O3, "emb-O3",
                   "Most optimizations plus expensive ones")),
    cl::desc("Optimization level for the embedded modules"),
    cl::cat(cl::catKitClDevOpts));

static cl::opt<bool> clPrintEmbPipelinePasses(
    "emb-print-pipeline-passes", cl::init(false), cl::Hidden,
    cl::desc("Print the passes that will be run on the embedded modules"),
    cl::cat(cl::catKitClDevOpts));

namespace {

/// Optimize the embedded bitcode. This runs the standard sequence of
/// optimization passes on it.
class EmbOptimize {
private:
  TTID tt;
  const TTOptions &tto;

protected:
  EmbOptimize(TTID tt, const TTOptions &tto) : tt(tt), tto(tto) {}

  /// Construct the pipeline tuning options. These may be different depending on
  /// the bitcode being optimized.
  virtual PipelineTuningOptions
  getPipelineTuningOptions(OptimizationLevel optLevel) = 0;

public:
  virtual ~EmbOptimize() = default;

  bool run(Module &devM) {
    auto getOptLevel = [](const TTOptions &tto) -> OptimizationLevel {
      // If the optimization level has been overridden on the command line,
      // prefer that, otherwise, use the optimization level from the TTOptions.
      if (clOptimizationLevel.getNumOccurrences())
        return clOptimizationLevel;
      return tto.getOptLevel();
    };

    // The analysis managers must be declared in this order so that they are
    // destroyed in the correct order due to inter-analysis-manager references
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;
    PassInstrumentationCallbacks pic;
    TargetMachine *tm = createTargetMachine(tt, tto);
    OptimizationLevel optLevel = getOptLevel(tto);
    PipelineTuningOptions pto = getPipelineTuningOptions(optLevel);

    // It is not clear why we need to disable the sandbox before creating a
    // pass builder. Should really figure it out.
    sys::sandbox::ScopedSetting disableSandbox = sys::sandbox::scopedDisable();
    PassBuilder pb(tm, pto);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    tm->registerPassBuilderCallbacks(pb);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(optLevel);
    mpm.addPass(VerifierPass());
    if (clPrintEmbPipelinePasses) {
      mpm.printPipeline(outs(), [&pic](StringRef className) -> StringRef {
        StringRef passName = pic.getPassNameForClassName(className);
        return passName.empty() ? className : passName;
      });
      outs() << "\n";
    }
    mpm.run(devM, mam);

    return true;
  }
};

/// Optimize a module for NVPTX.
class EmbOptimizeCuda : public EmbOptimize {
protected:
  PipelineTuningOptions
  getPipelineTuningOptions(OptimizationLevel optLevel) override final {
    unsigned speedupLevel = (unsigned)optLevel;
    PipelineTuningOptions pto;
    pto.LoopUnrolling = speedupLevel > 1;
    pto.LoopInterleaving = speedupLevel > 2;
    pto.LoopStripmine = speedupLevel > 2;
    pto.LoopVectorization = false;
    pto.SLPVectorization = false;

    return pto;
  }

public:
  EmbOptimizeCuda(const TTOptions &tto) : EmbOptimize(TTID::Cuda, tto) {}
};

/// Optimize a module for AMDGPU.
class EmbOptimizeHip : public EmbOptimize {
protected:
  PipelineTuningOptions
  getPipelineTuningOptions(OptimizationLevel optLevel) override final {
    unsigned speedupLevel = (unsigned)optLevel;
    PipelineTuningOptions pto;
    pto.LoopUnrolling = speedupLevel > 1;
    pto.LoopInterleaving = speedupLevel > 2;
    pto.LoopStripmine = speedupLevel > 2;
    pto.LoopVectorization = false;
    pto.SLPVectorization = false;

    return pto;
  }

public:
  EmbOptimizeHip(const TTOptions &tto) : EmbOptimize(TTID::Hip, tto) {}
};

} // namespace

bool EmbOptimizePass::run(TTID tt, Module &devM, Module &hostM,
                          ModuleAnalysisManager &hostMAM) {
  switch (tt) {
  case TTID::Cuda: return EmbOptimizeCuda(tto).run(devM);
  case TTID::Hip: return EmbOptimizeHip(tto).run(devM);
  default: llvm_unreachable("EmbOptimizePass::run: TTID not handled");
  }
}
