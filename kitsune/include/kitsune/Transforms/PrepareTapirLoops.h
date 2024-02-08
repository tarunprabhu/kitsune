//===- PrepareTapirLoops.h - Prepare tapir loops for lowering ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform tapir loops to a form suitable for parallel execution.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_PREPARE_TAPIR_LOOPS_H
#define KITSUNE_TRANSFORMS_PREPARE_TAPIR_LOOPS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Transform tapir loops to a form suitable for parallel execution.
class PrepareTapirLoopsPass : public PassInfoMixin<PrepareTapirLoopsPass> {
public:
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &am);
};

} // end namespace llvm

#endif // KITSUNE_TRANSFORMS_PREPARE_TAPIR_LOOPS_H
