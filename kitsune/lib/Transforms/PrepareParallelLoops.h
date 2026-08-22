//=- PrepareParallelLoops.h - Transform non-reduction tapir loops -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transform tapir loops that do not perform reductions to a form that is
// suitable for parallel execution.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_TRANSFORMS_PREPARE_PARALLEL_LOOPS_H
#define KITSUNE_LIB_TRANSFORMS_PREPARE_PARALLEL_LOOPS_H

namespace llvm {

class DominatorTree;
class LoopInfo;
class MemorySSA;
class ScalarEvolution;
class TapirLoopInfo;
class TaskInfo;

namespace detail {

/// Check that the given non-reduction tapir loop can be transformed to a form
/// that is suitable for parallel execution.
bool checkParallelLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                       LoopInfo &li);

/// Transform the given non-reduction tapir loop to a form that is suitable for
/// parallel execution.
bool prepareParallelLoop(TapirLoopInfo &tapirLoop, DominatorTree &dt,
                         LoopInfo &li, MemorySSA &mssa, ScalarEvolution &se,
                         TaskInfo &ti);

} // namespace detail

} // end namespace llvm

#endif // KITSUNE_LIB_TRANSFORMS_PREPARE_PARALLEL_LOOPS_H
