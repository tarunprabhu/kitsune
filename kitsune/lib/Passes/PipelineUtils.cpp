//===- PipelineUtils.h - Utilities to populate pass pipelines --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to construct Kitsune's pass pipelines.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Passes/PipelineUtils.h"
#include "kitsune/Analysis/PreLowerVerification.h"
#include "kitsune/Analysis/TTObjectsAnalysis.h"
#include "kitsune/CodeGen/CodeGenFatBinaries.h"
#include "kitsune/CodeGen/EmbLowerKitIntrinsics.h"
#include "kitsune/CodeGen/LowerKitIntrinsics.h"
#include "kitsune/CodeGen/StripKitAddrSpaces.h"
#include "kitsune/Transforms/DeLICM.h"
#include "kitsune/Transforms/EmbLinkLibDeviceBitcode.h"
#include "kitsune/Transforms/EmbLowerKitIntrinsicsEarly.h"
#include "kitsune/Transforms/EmbOptimize.h"
#include "kitsune/Transforms/EmbPrepare.h"
#include "kitsune/Transforms/EmbResolveLibDeviceCalls.h"
#include "kitsune/Transforms/GenerateCtors.h"
#include "kitsune/Transforms/LowerKitReduceIntrinsics.h"
#include "kitsune/Transforms/PreLowerAnnotate.h"
#include "kitsune/Transforms/PrefetchForDevice.h"
#include "kitsune/Transforms/PrepareReductionLoops.h"
#include "kitsune/Transforms/RecomputeKernelProperties.h"
#include "kitsune/Transforms/SecondaryIVElimination.h"
#include "kitsune/Transforms/Serialize.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/BDCE.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"

using namespace llvm;

bool llvm::isKitsuneOrTapirPipelineAlias(StringRef name) {
  // The tapir and kitsune pipeline aliases are parameterized i.e. they take
  // arguments. These are attached to the name of the pipeline itself e.g.
  // 'kit-lowering<O1>', 'tapir-lowering-loops<O2>' etc. This function may be
  // called with just the name, or it may be called with the parameters
  // attached. Do the right thing in both cases.
  StringRef baseName = name.take_while([](char c) { return c != '<'; });
  return baseName == "kit-lowering" || baseName == "tapir-lowering" ||
         baseName == "tapir-lowering-loops";
}

bool llvm::runKitNonLoweringPasses(ThinOrFullLTOPhase phase,
                                   const PipelineTuningOptions &pto) {
  // For now, we always run the passes not related to tapir-lowering as long as
  // a valid tapir target has been provided. When using LTO, this will run both
  // before and after the bitcode has been linked. It is possible that we only
  // need to run such passes during the prelink phase, but there is no harm in
  // running them both times.
  return pto.TTOpts.has_value();
}

bool llvm::runTapirLoweringPasses(ThinOrFullLTOPhase phase,
                                  const PipelineTuningOptions &pto) {
  // The tapir lowering passes should only be run as part of the post-link
  // pipeline.
  //
  // One big reason to use LTO with tapir is to resolve cross-translation-unit
  // references, especially with GPU tapir targets. Functions defined in a
  // different translation unit have to be added to the embedded bitcode modules
  // before it is compiled to GPU code.
  //
  if (not pto.TTOpts)
    return false;
  else if (pto.TTOpts->getTTID() == TTID::Nolo)
    return false;
  else if (phase == ThinOrFullLTOPhase::FullLTOPreLink or
           phase == ThinOrFullLTOPhase::ThinLTOPreLink)
    return false;
  else
    return true;
}

template <typename Pass, typename... Args>
static void addModulePass(ModulePassManager &mpm, Args &&...args) {
  mpm.addPass(Pass(args...));
}

template <typename Pass, typename... Args>
static void addFunctionPass(ModulePassManager &mpm, Args &&...args) {
  FunctionPassManager fpm;

  fpm.addPass(Pass(args...));
  mpm.addPass(createModuleToFunctionPassAdaptor(std::move(fpm)));
}

template <typename Pass, typename... Args>
static void addLoopPass(ModulePassManager &mpm, Args &&...args) {
  LoopPassManager lpm;
  FunctionPassManager fpm;

  lpm.addPass(Pass(args...));
  fpm.addPass(createFunctionToLoopPassAdaptor(std::move(lpm)));
  mpm.addPass(createModuleToFunctionPassAdaptor(std::move(fpm)));
}

ModulePassManager
llvm::populateKitPreTapirPasses(PassBuilder &pb, OptimizationLevel optLevel,
                                ThinOrFullLTOPhase ltoPhase,
                                const PipelineTuningOptions &pto) {
  ModulePassManager mpm;

  pb.invokeKitsunePreTapirEarlyEPCallbacks(mpm, optLevel);

  // There are currently no standard passes that are run before the tapir
  // lowering pipeline.

  pb.invokeKitsunePreTapirLateEPCallbacks(mpm, optLevel);

  return mpm;
}

static void populateSimplifyPasses(ModulePassManager &mpm) {
  // the kit-reductions pass may not leave the IR in as clean a state as we
  // would like. All these passes are probably overkill, but we definitely
  // need at least indvars and simplifycfg.
  //
  // FIXME: Some of these passes were added because an initial cut of the
  // implementation used Tapir's loop stripmining pass, and comments in the code
  // suggested that these would be useful. We have since used a totally new
  // approach, so it is not clear how many of these are actually needed.
  //
  addFunctionPass<EarlyCSEPass>(mpm, /*UseMemorySSA=*/true);
  addLoopPass<IndVarSimplifyPass>(mpm);
  addFunctionPass<SimplifyCFGPass>(mpm);
  addFunctionPass<InstCombinePass>(mpm);
  addFunctionPass<SCCPPass>(mpm);
  addFunctionPass<BDCEPass>(mpm);
  addFunctionPass<InstCombinePass>(mpm);
  addFunctionPass<DSEPass>(mpm);
  addFunctionPass<ADCEPass>(mpm);
}

ModulePassManager llvm::populateKitPreLoopSpawningPasses(
    PassBuilder &pb, OptimizationLevel optLevel, ThinOrFullLTOPhase ltoPhase,
    const PipelineTuningOptions &pto) {
  ModulePassManager mpm;

  // At optimization level O0, loop spawning will not be run, so there is no
  // point in running the other Kitsune-specific passes.
  if (optLevel.getSpeedupLevel() > 0) {
    addFunctionPass<LoopSimplifyPass>(mpm);
    addLoopPass<LoopRotatePass>(mpm);
    addLoopPass<SecondaryIVEliminationPass>(mpm);

    // SecondaryIVElimination will preserve the loop-simplify and loop-rotate
    // nature of the loop. It is not clear why we need to recompute LCSSA.
    addFunctionPass<LCSSAPass>(mpm);
    addFunctionPass<PrepareReductionLoopsPass>(mpm);
    addFunctionPass<LowerKitReduceIntrinsicsPass>(mpm);

    // We must run the module inliner because the reducer function should be
    // inlined after the loop has been prepared.
    addModulePass<ModuleInlinerPass>(mpm);

    // After preparing the loop for parallel reductions, we run the module
    // inliner. Between those two passes, the IR should be cleaned up.
    populateSimplifyPasses(mpm);

    // It is not clear if we really need to run loop-simplify here, but DeLICM
    // requires it, so we might as well.
    addFunctionPass<LoopSimplifyPass>(mpm);
    addFunctionPass<DeLICMPass>(mpm);

    // Run simplifycfg after the DeLICM pass since it may leave empty basic
    // blocks around.
    addFunctionPass<SimplifyCFGPass>(mpm);

    // Running simplifycfg may require the loop to be simplified again.
    // The PreLowerVerification pass requires the tapir loops to be in simplify
    // form.
    addFunctionPass<LoopSimplifyPass>(mpm);
    addModulePass<PreLowerVerificationPass>(mpm);

    // TODO:? Do we need to run the pre-lower verification pass after the
    // serialize pass?
    addFunctionPass<PreLowerAnnotatePass>(mpm);
    addFunctionPass<SerializePass>(mpm);
  }

  return mpm;
}

ModulePassManager
llvm::populateKitPostTapirPasses(PassBuilder &pb, OptimizationLevel optLevel,
                                 ThinOrFullLTOPhase ltoPhase,
                                 const PipelineTuningOptions &pto) {
  ModulePassManager mpm;

  // At optimization level O0, loop spawning will not have been run, so there
  // is no reason to run the passes that operate on the code generated by it.
  if (optLevel.getSpeedupLevel() > 0) {
    pb.invokeKitsunePostTapirEarlyEPCallbacks(mpm, optLevel);

    addModulePass<PrefetchForDevicePass>(mpm);
    addModulePass<EmbLowerKitIntrinsicsEarlyPass>(mpm);
    addModulePass<EmbResolveLibDeviceCallsPass>(mpm);
    addModulePass<EmbPreparePass>(mpm);
    addModulePass<EmbLinkLibDeviceBitcodePass>(mpm);
    addModulePass<EmbOptimizePass>(mpm);

    pb.invokeKitsunePostTapirLateEPCallbacks(mpm, optLevel);

    addModulePass<RecomputeKernelPropertiesPass>(mpm);
    addModulePass<GenerateCtorsPass>(mpm);

    pb.invokeKitsunePostTapirLastEPCallbacks(mpm, optLevel);
  }

  return mpm;
}

void llvm::populateKitCodeGenPasses(legacy::PassManager &pm,
                                    std::optional<TTOptions> tto) {
  if (tto) {
    pm.add(createTTObjectsAnalysisWrapperPass(tto));
    pm.add(createEmbLowerKitIntrinsicsLegacyPass());
    pm.add(createLowerKitIntrinsicsLegacyPass());
    pm.add(createStripKitAddrSpacesLegacyPass());
    pm.add(createCodeGenFatBinariesLegacyPass());
  }
}
