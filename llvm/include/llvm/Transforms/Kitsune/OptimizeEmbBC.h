//===- OptimizeEmbBC.h - Optimize embedded bitcode -------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Run the standard sequence of optimization passes on the embedded bitcode
// in the module.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_KITSUNE_OPTIMIZE_EMB_BC_H
#define LLVM_TRANSFORMS_KITSUNE_OPTIMIZE_EMB_BC_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// Some tapir targets embed bitcode into the module. For instance, the cuda and
/// hip tapir targets add bitcode that will eventually be compiled to binaries
/// to run on the GPU. Those tapir targets may generate multiple bitcode modules
/// - one for each kernel that will eventually be launched for execution on the
/// GPU. This will combine these into a single module per device architecture
/// and run the standard optimization passes on the result.
class OptimizeEmbBCPass : public PassInfoMixin<OptimizeEmbBCPass> {
public:
  PreservedAnalyses run(Module &m, ModuleAnalysisManager &mam);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_KITSUNE_OPTIMIZE_EMB_BC_H
