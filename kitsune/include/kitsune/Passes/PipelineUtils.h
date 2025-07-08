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

#include "kitsune/Core/TapirTargetOptions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"

#include <optional>

namespace llvm {

class PipelineTuningOptions;

/// Check if the tapir (and by extension Kitsune) lowering pipeline should be
/// used.
bool useTapirLowering(ThinOrFullLTOPhase phase,
                      const PipelineTuningOptions &pto);

/// Populate a ModulePassManager with the passes that should be run as part of
/// Kitsune's pre-tapir pipeline. These passes are run immediately before tapir
/// lowering.
ModulePassManager populateKitPreTapirPasses(OptimizationLevel level,
                                            ThinOrFullLTOPhase phase,
                                            const PipelineTuningOptions &pto);

/// Populate a ModulePassManager with the passes that should be run as part of
/// Kitsune's post-tapir pipeline. These passes are run immediately after tapir
/// lowering.
ModulePassManager populateKitPostTapirPasses(OptimizationLevel level,
                                             ThinOrFullLTOPhase phase,
                                             const PipelineTuningOptions &pto);

/// Populate a pass manager with Kitsune's codegen passes.
void populateKitCodeGenPasses(legacy::PassManager &pm,
                              std::optional<TapirTargetOptions> tto);

} // namespace llvm

#endif // KITSUNE_PASSES_PIPELINE_UTILS_H
