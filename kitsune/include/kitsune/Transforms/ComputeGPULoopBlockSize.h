//===- ComputeGPULoopBlockSize.h - Iterations per GPU thread ---*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compute the number of iterations of a tapir loop to execute on each GPU
// thread.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TRANSFORMS_COMPUTE_GPU_LOOP_BLOCK_SIZE_H
#define KITSUNE_TRANSFORMS_COMPUTE_GPU_LOOP_BLOCK_SIZE_H

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class LPMUpdater;

/// \ingroup kitsune
/// Compute the number of iterations of a tapir loop to execute on each GPU
/// thread and lower calls to the kit.gpu.ipt intrinsic with this value.
class ComputeGPULoopBlockSizePass
    : public PassInfoMixin<ComputeGPULoopBlockSizePass> {
public:
  PreservedAnalyses run(Loop &loop, LoopAnalysisManager &am,
                        LoopStandardAnalysisResults &ar, LPMUpdater &updater);
};

} // namespace llvm

#endif // KITSUNE_TRANSFORMS_COMPUTE_GPU_LOOP_BLOCK_SIZE_H
