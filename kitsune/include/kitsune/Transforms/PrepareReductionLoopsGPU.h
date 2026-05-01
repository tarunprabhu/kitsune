//- PrepareReductionLoopsGPU.h - Transform reduction loops for GPU -*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass that transforms tapir loops that perform reductions to a form that is
// suitable for parallel execution on GPU's.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_PREPARE_REDUCTION_LOOPS_GPU_H
#define KITSUNE_TRANSFORMS_PREPARE_REDUCTION_LOOPS_GPU_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// \ingroup kitsune
/// Transform tapir loops that perform reductions to a form that is suitable for
/// execution on GPU's.
///
/// TODO: Write more documentation describing what the loops will look like
/// before and after the transformation.
///
class PrepareReductionLoopsGPUPass
    : public PassInfoMixin<PrepareReductionLoopsGPUPass> {
public:
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &am);
};

} // end namespace llvm

#endif // KITSUNE_TRANSFORMS_PREPARE_REDUCTION_LOOPS_GPU_H
