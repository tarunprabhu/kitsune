//==- PrepareReductionLoops.h - Transform tapir reduction loops -*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform tapir loops that perform reductions to a form that is suitable for
// parallel execution.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_TRANSFORMS_PREPARE_REDUCTION_LOOPS_H
#define KITSUNE_LIB_TRANSFORMS_PREPARE_REDUCTION_LOOPS_H

#include "kitsune/Core/Reductions.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class DominatorTree;
class Loop;
class LoopInfo;
class MemorySSA;
class ScalarEvolution;
class TapirLoopInfo;
class TaskInfo;

namespace detail {

/// Check that the given tapir reduction loop can be transformed to a form that
/// is suitable for parallel execution.
bool checkReductionLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                        LoopInfo &li);

/// Transform the given tapir reduction loop to a form that is suitable for
/// parallel execution.
bool prepareReductionLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                          LoopInfo &li, MemorySSA &mssa, ScalarEvolution &se,
                          TaskInfo &ti);

bool prepareReductionLoopForCPU(
    TapirLoopInfo &tapirLoop, const SmallVectorImpl<ReductionInfo> &reductions,
    DominatorTree &dt, LoopInfo &li, MemorySSA &mssa, ScalarEvolution &se,
    TaskInfo &ti);

bool prepareReductionLoopForGPU(
    Loop &loop, const SmallVectorImpl<ReductionInfo> &reductions);

} // namespace detail

} // end namespace llvm

#endif // KITSUNE_LIB_TRANSFORMS_PREPARE_REDUCTION_LOOPS_H
