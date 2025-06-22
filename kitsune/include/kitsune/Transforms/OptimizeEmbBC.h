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

#ifndef KITSUNE_TRANSFORMS_OPTIMIZE_EMB_BC_H
#define KITSUNE_TRANSFORMS_OPTIMIZE_EMB_BC_H

#include "kitsune/Transforms/EmbBCPass.h"

namespace llvm {

/// This will run the standard optimization passes on the embedded module for
/// the given tapir target.
class OptimizeEmbBCPass : public EmbBCPass<OptimizeEmbBCPass> {
public:
  bool run(TTID tt, Module &devM, Module &hostM, ModuleAnalysisManager &hostAM);

  using EmbBCPass<OptimizeEmbBCPass>::run;
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_OPTIMIZE_EMB_BC_H
