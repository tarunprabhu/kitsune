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

#include "llvm/Transforms/Kitsune/EmbBCPass.h"

namespace llvm {

/// This will run the standard optimization passes on the embedded module for
/// the given tapir target.
class OptimizeEmbBCPass : public EmbBCPass<OptimizeEmbBCPass> {
public:
  bool run(TapirTargetID tt, Module &devM, Module &hostM,
           ModuleAnalysisManager &hostMAM);

  using EmbBCPass<OptimizeEmbBCPass>::run;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_KITSUNE_OPTIMIZE_EMB_BC_H
