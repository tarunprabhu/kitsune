//===- FinalizeKernelMetadata.h - Compute kernel metadata ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compute the kernel metadata used in kernel launches.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_KITSUNE_FINALIZE_KERNEL_METADATA_H
#define LLVM_TRANSFORMS_KITSUNE_FINALIZE_KERNEL_METADATA_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// Some metadata about the kernel being launched is passed to the launch calls.
/// These may be used to determine launch parameters. Currently, this metadata
/// includes information about the instruction mix within the kernel - the
/// numbers of floating point, integer and memory operations in the kernel's IR.
/// In the future, this could be expanded to include anything else that could be
/// useful. This pass is run as late in the pipeline as possible to allow for
/// all the optimizations to be run on the embedded bitcode.
class FinalizeKernelMetadataPass
    : public PassInfoMixin<FinalizeKernelMetadataPass> {
public:
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_KITSUNE_FINALIZE_KERNEL_METADATA_H
