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

namespace llvm {

class DominatorTree;
class LoopInfo;
class MemorySSA;
class ScalarEvolution;
class TapirLoopInfo;
class TaskInfo;

/// Check that the given tapir reduction loop can be transformed to a form that
/// is suitable for parallel execution.
bool checkReductionLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                        LoopInfo &li);

/// Transform the given tapir reduction loop to a form that is suitable for
/// parallel execution.
bool prepareReductionLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                          LoopInfo &li, MemorySSA &mssa, ScalarEvolution &se,
                          TaskInfo &ti);

} // end namespace llvm

#endif // KITSUNE_LIB_TRANSFORMS_PREPARE_REDUCTION_LOOPS_H
