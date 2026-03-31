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
#include "kitsune/Transforms/EmbLinkLibDeviceBitcode.h"
#include "kitsune/Transforms/EmbLowerKitIntrinsicsLibDevice.h"
#include "kitsune/Transforms/EmbOptimize.h"
#include "kitsune/Transforms/EmbPrepare.h"
#include "kitsune/Transforms/EmbResolveLibDeviceCalls.h"
#include "kitsune/Transforms/GenerateCtors.h"
#include "kitsune/Transforms/PreLowerAnnotate.h"
#include "kitsune/Transforms/PrefetchForDevice.h"
#include "kitsune/Transforms/RecomputeKernelProperties.h"
#include "kitsune/Transforms/Serialize.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
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

ModulePassManager llvm::populateKitPreLoopSpawningPasses(
    PassBuilder &pb, OptimizationLevel optLevel, ThinOrFullLTOPhase ltoPhase,
    const PipelineTuningOptions &pto) {
  ModulePassManager mpm;

  // At optimization level O0,, loop spawning will not be run, so there is no
  // point in running the other Kitsune-specific passes.
  if (optLevel.getSpeedupLevel() > 0) {
    mpm.addPass(PreLowerVerificationPass());

    // annotate-tapir-loops requires the loops to be in simplified and rotated
    // form. Since this pipeline is run just before loop-spawning, both the
    // loop-simplify and loop-rotate passes are guaranteed to have been run.
    mpm.addPass(PreLowerAnnotatePass());
    mpm.addPass(SerializePass());
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

    mpm.addPass(PrefetchForDevicePass());
    mpm.addPass(EmbLowerKitIntrinsicsLibDevicePass());
    mpm.addPass(EmbResolveLibDeviceCallsPass());
    mpm.addPass(EmbPreparePass());
    mpm.addPass(EmbLinkLibDeviceBitcodePass());
    mpm.addPass(EmbOptimizePass());

    pb.invokeKitsunePostTapirLateEPCallbacks(mpm, optLevel);

    mpm.addPass(RecomputeKernelPropertiesPass());
    mpm.addPass(GenerateCtorsPass());

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
