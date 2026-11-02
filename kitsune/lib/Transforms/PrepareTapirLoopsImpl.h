//==- PrepareTapirLoopsImpl.h - Prepare tapir loops for lowering -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private header for the implementation of the kit-prepare pass that prepares
// tapir loops for lowering.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_LIB_TRANSFORMS_PREPARE_TAPIR_LOOPS_IMPL_H
#define KITSUNE_LIB_TRANSFORMS_PREPARE_TAPIR_LOOPS_IMPL_H

namespace llvm {

class DominatorTree;
class Loop;
class LoopInfo;
class MemorySSA;
class ScalarEvolution;
class TapirLoopInfo;
class TaskInfo;

namespace detail {

bool prepareTapirNestCPU(TapirLoopInfo &root, DominatorTree &dt, LoopInfo &li,
                         MemorySSA &mssa, ScalarEvolution &se, TaskInfo &ti);

bool prepareTapirNestGPU(Loop &root);

} // namespace detail

} // namespace llvm

#endif // KITSUNE_LIB_TRANSFORMS_PREPARE_TAPIR_LOOPS_IMPL_H
