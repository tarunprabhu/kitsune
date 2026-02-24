//===- QthreadsTT.h - Tapir target that lowers to qthreads -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tapir target that lowers to qthreads.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TARGETS_QTHREADS_TT_H
#define KITSUNE_TARGETS_QTHREADS_TT_H

#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {

class TTOptions;

/// Tapir target that lowers to qthreads via a thin wrapper provided by Kitsune.
/// The underlying qthreads runtime determines how to split the iterations of
/// a parallel loop across available compute elements ("shepherds" in qthreads
/// terminology).
/// \ingroup kitsune
class QthreadsTT : public TapirTarget {
public:
  QthreadsTT(Module &m, const TTOptions &ttOpts);
  virtual ~QthreadsTT() = default;

  /// Lower a call to the tapir.loop.grainsize intrinsic into a grainsize
  /// (coarsening) value.
  Value *lowerGrainsizeCall(CallInst *call) override final;

  /// Lower a Tapir sync instruction \p si.
  void lowerSync(SyncInst &si) override final;

  /// Returns true if tasks in Function \p f should be outlined into their own
  /// functions.
  bool shouldDoOutlining(const Function &f) const override final;

  /// Process function \p f before any function outlining is performed. This
  /// routine should not modify the CFG structure, unless it processes all Tapir
  /// instructions in \p F itself. Returns true if it modifies the CFG, false
  /// otherwise.
  bool preProcessFunction(Function &f, TaskInfo &ti,
                          bool processingTapirLoops) override final {
    return false;
  }

  /// Process function \p f at the end of the lowering process.
  void postProcessFunction(Function &f,
                           bool processingTapirLoops) override final {
    // Nothing to be done here
  }

  /// Process a generated helper function \p f produced via outlining, at the
  /// end of the lowering process.
  void postProcessHelper(Function &f) override final {
    // Nothing to be done here
  }

  /// Pre-process the function \p f that has just been outlined from a task.
  /// This routine is executed on each outlined function by traversing in
  /// post-order the tasks in the original function.
  void preProcessOutlinedTask(Function &f, Instruction *detachPt,
                              Instruction *tfCreate, bool isSpawner,
                              BasicBlock *tfEntry) override final {
    // Nothing to be done here
  }

  /// Post-process the function \p f that has just been outlined from a task.
  /// This routine is executed on each outlined function by traversing in
  /// post-order the tasks in the original function.
  void postProcessOutlinedTask(Function &f, Instruction *detachPt,
                               Instruction *tfCreate, bool isSpawner,
                               BasicBlock *tfEntry) override final {
    // Nothing to be done here
  }

  /// Pre-process the root function \p f as a function that can spawn subtasks.
  void preProcessRootSpawner(Function &f, BasicBlock *tfEntry) override final {
    // Nothing to do here because none of the functions processed by this tapir
    // target can spawn subtasks.
  }

  /// Post-process the root Function \p F as a function that can spawn subtasks.
  void postProcessRootSpawner(Function &f, BasicBlock *tfEntry) override final {
    // Nothing to do here because none of the functions processed by this tapir
    // target can spawn subtasks.
  }

  /// Process the invocation of a task for an outlined function. This routine
  /// is invoked after processSpawner once for each child subtask.
  void processSubTaskCall(TaskOutlineInfo &toi,
                          DominatorTree &dt) override final {
    // Nothing to do here since nothing handled by this tapir target spawns
    // subtasks.
  }

  /// Create a custom loop outline processor for this tapir target.
  LoopOutlineProcessor *
  getLoopOutlineProcessor(const TapirLoopInfo *tl) override final;
};

} // namespace llvm

#endif // KITSUNE_TARGETS_QTHREADS_TT_H
