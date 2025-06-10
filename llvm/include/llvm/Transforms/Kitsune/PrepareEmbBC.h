//=- PrepareEmbBC.h - Prepare the embedded bitcode for codegen ---*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prepare the embedded bitcode for code generation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_KITSUNE_PREPARE_EMB_BC_H
#define LLVM_TRANSFORMS_KITSUNE_PREPARE_EMB_BC_H

#include "llvm/Transforms/Kitsune/EmbBCPass.h"

namespace llvm {

/// Prepare the embedded bitcode for code generation. This should be run
/// relatively late in the pipeline. This will carry out any
/// architecture-specific transformations that are unrelated to optimizations
/// and that have not already been carried out by the tapir targets that created
/// this bitcode. For instance, for AMDGPU kernels, the kernel function
/// arguments must be placed in a specific address space, as must any alloca's
/// in the kernel functions. In some cases, the calling conventions of such
/// functions must also be changed.
class PrepareEmbBCPass : public EmbBCPass<PrepareEmbBCPass> {
public:
  bool run(TapirTargetID tt, Module &km, Module &hostM,
           ModuleAnalysisManager &hostMAM);

  using EmbBCPass<PrepareEmbBCPass>::run;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_KITSUNE_PREPARE_EMB_BC_H
