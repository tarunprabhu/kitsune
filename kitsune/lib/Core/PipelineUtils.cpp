//===- PipelineUtils.h - Utilities to populate pass pipelines --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities related to Kitsune's address spaces.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/PipelineUtils.h"
#include "kitsune/Analysis/TapirTargetAnalysis.h"
#include "kitsune/CodeGen/CodeGenFatBinaries.h"
#include "kitsune/CodeGen/LowerKitsuneIntrinsics.h"
#include "kitsune/CodeGen/StripKitsuneAddrSpaces.h"

using namespace llvm;

void llvm::populateKitPreTapirPasses(ModulePassManager &mpm,
                                     OptimizationLevel level,
                                     ThinOrFullLTOPhase phase,
                                     const PipelineTuningOptions &pto) {}

void llvm::populateKitPostTapirPasses(ModulePassManager &mpm,
                                      OptimizationLevel level,
                                      ThinOrFullLTOPhase phase,
                                      const PipelineTuningOptions &pto) {}

void llvm::populateKitCodeGenPasses(legacy::PassManager &pm,
                                    std::optional<TapirTargetOptions> tto) {
  pm.add(createTapirTargetAnalysisWrapperPass(tto));
  pm.add(createLowerKitsuneIntrinsicsLegacyPass());
  pm.add(createStripKitsuneAddrSpacesLegacyPass());
  pm.add(createCodeGenFatBinariesLegacyPass());
}
