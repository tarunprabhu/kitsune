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

#ifndef KITSUNE_PASSES_PIPELINE_UTILS_H
#define KITSUNE_PASSES_PIPELINE_UTILS_H

#include "kitsune/Core/TTOptions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"

#include <optional>

namespace llvm {

/// \addtogroup kitsune
/// @{

class PassBuilder;
class PipelineTuningOptions;

/// Is the pass name a Tapir/Kitsune lowering pipeline alias.
bool isKitsuneOrTapirPipelineAlias(StringRef name);

/// Check if the Kitsune's early verification pass should be run. This typically
/// runs early in the pipeline and is intended to catch errors that the frontend
/// missed.
bool runKitEarlyVerificationPasses(ThinOrFullLTOPhase phase,
                                   const PipelineTuningOptions &pto);

/// Check if the Kitsune-specific passes that prepare the code for tapir
/// lowering should be run.
bool runKitPreparePasses(ThinOrFullLTOPhase phase,
                         const PipelineTuningOptions &pto);

/// Check if the passes that are part of the tapir (and by extension Kitsune)
/// lowering pipeline should be run.
bool runTapirLoweringPasses(ThinOrFullLTOPhase phase,
                            const PipelineTuningOptions &pto);

/// Populate a ModulePassManager with passes that should be run early in the
/// optimization pipeline.
FunctionPassManager populateKitEarlyPasses(PassBuilder &pb,
                                           OptimizationLevel optLevel,
                                           ThinOrFullLTOPhase ltoPhase,
                                           const PipelineTuningOptions &opts);

/// Populate a ModulePassManager with Kitsune's early verification passes.
ModulePassManager
populateKitEarlyVerificationPasses(PassBuilder &pb, OptimizationLevel optLevel,
                                   ThinOrFullLTOPhase ltoPhase,
                                   const PipelineTuningOptions &opts);

/// Populate a ModulePassManager with passes that prepare tapir loops for
/// lowering. These passes are generally run before the vectorized, and are,
/// therefore, not part of Kitsune's lowering pipeline.
ModulePassManager populateKitPreparePasses(PassBuilder &pb,
                                           OptimizationLevel optLevel,
                                           ThinOrFullLTOPhase ltoPhase,
                                           const PipelineTuningOptions &pto);

/// Populate a ModulePassManager with the passes that should be run as part of
/// Kitsune's pre-tapir pipeline. These passes are run immediately before tapir
/// lowering.
ModulePassManager populateKitPreTapirPasses(PassBuilder &pb,
                                            OptimizationLevel optLevel,
                                            ThinOrFullLTOPhase ltoPhase,
                                            const PipelineTuningOptions &pto);

/// Populate a ModulePassManager with the passes that should be run as part of
/// Kitsune's pre-loop-spawning pipeline. These passes are run immediately
/// before the loop-spawning pass is run.
ModulePassManager
populateKitPreLoopSpawningPasses(PassBuilder &pb, OptimizationLevel optLevel,
                                 ThinOrFullLTOPhase ltoPhase,
                                 const PipelineTuningOptions &pto);

/// Populate a ModulePassManager with the passes that should be run as part of
/// Kitusne's post-loop-spawning pipeline. After loop-spawning, the standard
/// function simplification pipeline is run. These passes are run *after* those
/// simplification passes have been run.
ModulePassManager
populateKitPostLoopSpawningPasses(PassBuilder &pb, OptimizationLevel optLevel,
                                  ThinOrFullLTOPhase ltoPhase,
                                  const PipelineTuningOptions &pto);

/// Populate a ModulePassManager with the passes that should be run as part of
/// Kitsune's post-tapir pipeline. These passes are run immediately after tapir
/// lowering.
ModulePassManager populateKitPostTapirPasses(PassBuilder &pb,
                                             OptimizationLevel optLevel,
                                             ThinOrFullLTOPhase ltoPhase,
                                             const PipelineTuningOptions &pto);

/// Populate a pass manager with Kitsune's codegen passes.
void populateKitCodeGenPasses(legacy::PassManager &pm,
                              std::optional<TTOptions> tto);

/// @}

} // namespace llvm

#endif // KITSUNE_PASSES_PIPELINE_UTILS_H
