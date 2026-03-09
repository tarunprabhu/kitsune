//==- RecomputeKernelProperties.h - Recompute kernel properties -*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Update the global variables containing the properties of functions launched
// in kernel launch calls.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_RECOMPUTE_KERNEL_PROPERTIES_H
#define KITSUNE_TRANSFORMS_RECOMPUTE_KERNEL_PROPERTIES_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Compute properties of kernels in embedded bitcode and update the
/// initializers of the corresponding global variables with these.
class RecomputeKernelPropertiesPass
    : public PassInfoMixin<RecomputeKernelPropertiesPass> {
public:
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &am);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_RECOMPUTE_KERNEL_PROPERTIES_H
