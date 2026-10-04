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

#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"

namespace llvm {

class BasicBlock;
class Function;
class Value;
class TapirLoopInfo;

namespace detail {

/// Base class to transform tapir loops for parallel execution.
class PrepareTapirLoopBase {
protected:
  DominatorTree &dt;
  LoopInfo &li;
  MemorySSA &mssa;
  ScalarEvolution &se;
  TaskInfo &ti;

  DomTreeUpdater dtu;
  MemorySSAUpdater mssau;

protected:
  void updateOuterLoopIV(Loop &outerLoop, Value *outerTC);
  void updateInnerLoopIV(Loop &outerLoop, Loop &innerLoop, Value *outerTC);
  void updateInnerLoopGuardCondition(Loop &innerLoop);
  void parallelizeOuterLoop(Loop &outerLoop, Loop &loop);
  void serializeInnerLoop(Loop &loop);

protected:
  virtual Value *computeOuterLoopTripCount(BasicBlock &bb,
                                           const TapirLoopInfo &tapirLoop) = 0;

  PrepareTapirLoopBase(DominatorTree &dt, LoopInfo &li, MemorySSA &mssa,
                       ScalarEvolution &se, TaskInfo &ti)
      : dt(dt), li(li), mssa(mssa), se(se), ti(ti),
        dtu(dt, DomTreeUpdater::UpdateStrategy::Eager), mssau(&mssa) {}

public:
  bool run(TapirLoopInfo &tapirLoop);
  virtual ~PrepareTapirLoopBase() = default;
};

/// Class to transform non-reduction tapir loops for parallel execution for
/// Kitsune's CPU-centric parallel runtimes.
class PrepareTapirLoopCPU : public PrepareTapirLoopBase {
protected:
  virtual Value *
  computeOuterLoopTripCount(BasicBlock &bb,
                            const TapirLoopInfo &tapirLoop) override;

public:
  PrepareTapirLoopCPU(DominatorTree &dt, LoopInfo &li, MemorySSA &mssa,
                      ScalarEvolution &se, TaskInfo &ti)
      : PrepareTapirLoopBase(dt, li, mssa, se, ti) {}
  virtual ~PrepareTapirLoopCPU() = default;
};

/// Class to transform non-reduction tapir loops for parallel execution on
/// GPU's.
class PrepareTapirLoopGPU : public PrepareTapirLoopBase {
protected:
  virtual Value *
  computeOuterLoopTripCount(BasicBlock &bb,
                            const TapirLoopInfo &tapirLoop) override;

public:
  PrepareTapirLoopGPU(DominatorTree &dt, LoopInfo &li, MemorySSA &mssa,
                      ScalarEvolution &se, TaskInfo &ti)
      : PrepareTapirLoopBase(dt, li, mssa, se, ti) {}
  virtual ~PrepareTapirLoopGPU() = default;
};

} // namespace detail

} // namespace llvm

#endif // KITSUNE_LIB_TRANSFORMS_PREPARE_TAPIR_LOOPS_IMPL_H
