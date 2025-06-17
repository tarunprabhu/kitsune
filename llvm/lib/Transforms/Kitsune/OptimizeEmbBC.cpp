//===- OptimizeEmbBC.cpp - Optimize embedded bitcode -----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Run the standard sequence of optimization passes on the embedded bitcode.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Kitsune/OptimizeEmbBC.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/TapirTargetAnalysis.h"
#include "llvm/Frontend/Tapir/OptLevelUtils.h"
#include "llvm/Frontend/Tapir/TapirTargetOptions.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/KitsuneMetadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Kitsune/EmbBCPassUtils.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Utils/KitsuneUtils.h"

#define DEBUG_TYPE "optimize-emb-bc"

using namespace llvm;

// Set a specific optimization level for the embedded bitcode. If this has not
// been set explicitly, the optimization level from the tapir target options
// will be used. This is typically set by the frontend.
//
// This is primarily useful for exploring various details of levels between
// those operating on the Tapir IR and those after the transformation to
// GPU-friendly LLVM IR.
static cl::opt<int>
    clOptLevel("emb-opt-level", cl::init(-1), cl::Hidden,
               cl::desc("Specify the embedded bitcode kernel optimization "
                        "level. Must be 0, 1, 2 or 3"));

namespace {

/// Optimize the embedded bitcode. This runs the standard sequence of
/// optimization passes on it.
class OptimizeModule {
private:
  TapirTargetID tt;
  const TapirTargetOptions &tto;

protected:
  OptimizeModule(TapirTargetID tt, const TapirTargetOptions &tto)
      : tt(tt), tto(tto) {}

  /// Construct the pipeline tuning options. These may be different depending on
  /// the bitcode being optimized.
  virtual PipelineTuningOptions
  getPipelineTuningOptions(OptimizationLevel optLevel) = 0;

public:
  virtual ~OptimizeModule() = default;

  bool run(Module &devM) {
    // If the optimization level has been overridden on the command line, prefer
    // that, otherwise, use the optimization level from the tapir target options
    OptimizationLevel optLevel = tto.getOptLevel();
    if (clOptLevel != -1)
      optLevel = mapToOptimizationLevel(clOptLevel);

    // If the speedup level is 0, no optimization passes are run.
    if (not optLevel.getSpeedupLevel())
      return false;

    // The analysis managers must be declared in this order so that they are
    // destroyed in the correct order due to inter-analysis-manager references
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;
    TargetMachine *tm = createTargetMachine(tt, tto);
    PipelineTuningOptions pto = getPipelineTuningOptions(optLevel);

    PassBuilder pb(tm, pto);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    tm->registerPassBuilderCallbacks(pb);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(optLevel);
    mpm.addPass(VerifierPass());
    mpm.run(devM, mam);

    return true;
  }
};

/// Optimize a module for NVPTX.
class OptimizeModuleCuda : public OptimizeModule {
protected:
  PipelineTuningOptions
  getPipelineTuningOptions(OptimizationLevel optLevel) override final {
    unsigned speedupLevel = optLevel.getSpeedupLevel();
    PipelineTuningOptions pto;
    pto.LoopUnrolling = speedupLevel > 1;
    pto.LoopInterleaving = speedupLevel > 2;
    pto.LoopStripmine = speedupLevel > 2;
    pto.LoopVectorization = false;
    pto.SLPVectorization = false;

    return pto;
  }

public:
  OptimizeModuleCuda(const TapirTargetOptions &tto)
      : OptimizeModule(TapirTargetID::Cuda, tto) {}
};

/// Optimize a module for AMDGPU.
class OptimizeModuleHip : public OptimizeModule {
protected:
  PipelineTuningOptions
  getPipelineTuningOptions(OptimizationLevel optLevel) override final {
    unsigned speedupLevel = optLevel.getSpeedupLevel();
    PipelineTuningOptions pto;
    pto.LoopUnrolling = speedupLevel > 1;
    pto.LoopInterleaving = speedupLevel > 2;
    pto.LoopStripmine = speedupLevel > 2;
    pto.LoopVectorization = false;
    pto.SLPVectorization = false;

    return pto;
  }

public:
  OptimizeModuleHip(const TapirTargetOptions &tto)
      : OptimizeModule(TapirTargetID::Hip, tto) {}
};

} // namespace

namespace llvm {

bool OptimizeEmbBCPass::run(TapirTargetID tt, Module &devM, Module &hostM,
                            ModuleAnalysisManager &hostMAM) {
  const TapirTargetInfo &tgi = hostMAM.getResult<TapirTargetAnalysis>(hostM);
  const TapirTargetOptions &tto = tgi.getOptions();

  switch (tt) {
  case TapirTargetID::Cuda:
    return OptimizeModuleCuda(tto).run(devM);
  case TapirTargetID::Hip:
    return OptimizeModuleHip(tto).run(devM);
  default:
    llvm_unreachable("OptimizeEmbBCPass::run: TapirTargetID not handled");
  }
}

} // namespace llvm
