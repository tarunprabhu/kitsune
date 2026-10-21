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

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"

namespace llvm {

class TTOptions;

class PassBuilder;
class PipelineTuningOptions;

/// \addtogroup kitsune
/// @{

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

/// Create a pipeline with passes that should be run early in the optimization
/// pipeline. These are generally run immediately after mem2reg (or the
/// equivalent).
FunctionPassManager populateKitEarlyPasses(PassBuilder &pb,
                                           OptimizationLevel optLevel,
                                           ThinOrFullLTOPhase ltoPhase,
                                           const PipelineTuningOptions &opts);

/// Create a pipeline with Kitsune's early verification passes.
ModulePassManager
populateKitEarlyVerificationPasses(PassBuilder &pb, OptimizationLevel optLevel,
                                   ThinOrFullLTOPhase ltoPhase,
                                   const PipelineTuningOptions &opts);

/// Create a pipeline with passes that prepare tapir loops for lowering. Several
/// of the passes that are run here need to identify loop guards, associate sync
/// instructions with loops and so on. This is more easily done early in the
/// optimization pipeline.
FunctionPassManager populateKitPreparePasses(PassBuilder &pb,
                                             OptimizationLevel optLevel,
                                             ThinOrFullLTOPhase ltoPhase,
                                             const PipelineTuningOptions &pto);

/// Create a pipeline with passes that run before loop-stripmine and the
/// vectorizer. This is the last chance to clean up code to maintain Kitsune's
/// invariants. The vectorizer breaks certain invariants that we would prefer to
/// maintain, such as adding instructions non-PHI instructions into a tapir loop
/// header. Ideally, we should modify the vectorizer to prevent it from doing
/// this, and run these as part of the pre-tapir or pre-loop-spawning pipelines,
/// but for now, we run these before the vectorizer.
ModulePassManager
populateKitPreVectorizePasses(PassBuilder &pb, OptimizationLevel optLevel,
                              ThinOrFullLTOPhase ltoPhase,
                              const PipelineTuningOptions &pto);

/// Populate a pass pipeline with the passes that should be run as part of
/// Kitsune's pre-tapir pipeline. These passes are run immediately before tapir
/// lowering.
ModulePassManager populateKitPreTapirPasses(PassBuilder &pb,
                                            OptimizationLevel optLevel,
                                            ThinOrFullLTOPhase ltoPhase,
                                            const PipelineTuningOptions &pto);

/// Create a pipeline with the passes that should be run as part of Kitsune's
/// pre-loop-spawning pipeline. These passes are run immediately before
/// loop-spawning.
ModulePassManager
populateKitPreLoopSpawningPasses(PassBuilder &pb, OptimizationLevel optLevel,
                                 ThinOrFullLTOPhase ltoPhase,
                                 const PipelineTuningOptions &pto);

/// Create a pipeline with the passes that should be run after loop-spawning.
/// The standard function simplification pipeline is run immediately after
/// loop-spawning. These passes are run *after* those simplification passes.
ModulePassManager
populateKitPostLoopSpawningPasses(PassBuilder &pb, OptimizationLevel optLevel,
                                  ThinOrFullLTOPhase ltoPhase,
                                  const PipelineTuningOptions &pto);

/// Create a pipeline with the passes that should be run as part of Kitsune's
/// post-tapir pipeline. These passes are run immediately after tapir lowering.
ModulePassManager populateKitPostTapirPasses(PassBuilder &pb,
                                             OptimizationLevel optLevel,
                                             ThinOrFullLTOPhase ltoPhase,
                                             const PipelineTuningOptions &pto);

/// Create a pipeline with Kitsune's codegen passes.
void populateKitCodeGenPasses(legacy::PassManager &pm, const TTOptions &tto);

/// @}

} // namespace llvm

#endif // KITSUNE_PASSES_PIPELINE_UTILS_H
