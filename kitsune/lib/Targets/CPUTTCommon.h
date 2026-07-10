//===- CPUTTCommon.h - Base class for CPU-centric tapir targets -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class for CPU-centric, threaded tapir targets for which the default
// lowering is sufficient.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TARGETS_CPUTT_COMMON_H
#define KITSUNE_TARGETS_CPUTT_COMMON_H

#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {

class TTOptions;

/// \addtogroup kitsune
/// \@{

/// Base class for CPU-centric, threaded tapir targets for which the default
/// lowering is sufficient. This assumes that functions do not spawn subtasks.
class CPUTTBase : public TapirTarget {
protected:
  CPUTTBase(Module &m, const TTOptions &ttOpts);

public:
  virtual ~CPUTTBase() = default;

  /// Lower a call to the tapir.loop.grainsize intrinsic into a grainsize
  /// (coarsening) value.
  Value *lowerGrainsizeCall(CallInst *call) override;

  /// Lower a Tapir sync instruction \p si.
  void lowerSync(SyncInst &si) override;

  /// Returns true if tasks in Function \p f should be outlined into their own
  /// functions.
  bool shouldDoOutlining(const Function &f) const override;

  /// Process function \p f before any function outlining is performed. This
  /// routine should not modify the CFG structure, unless it processes all Tapir
  /// instructions in \p f itself. Returns true if it modifies the CFG, false
  /// otherwise.
  bool preProcessFunction(Function &f, TaskInfo &ti,
                          bool processingTapirLoops) override;

  /// Process function \p f at the end of the lowering process.
  void postProcessFunction(Function &f, bool processingTapirLoops) override;

  /// Process a generated helper function \p f produced via outlining, at the
  /// end of the lowering process.
  void postProcessHelper(Function &f) override;

  /// Pre-process the function \p f that has just been outlined from a task.
  /// This routine is executed on each outlined function by traversing in
  /// post-order the tasks in the original function.
  void preProcessOutlinedTask(Function &f, Instruction *detachPt,
                              Instruction *tfCreate, bool isSpawner,
                              BasicBlock *tfEntry) override;

  /// Post-process the function \p f that has just been outlined from a task.
  /// This routine is executed on each outlined function by traversing in
  /// post-order the tasks in the original function.
  void postProcessOutlinedTask(Function &f, Instruction *detachPt,
                               Instruction *tfCreate, bool isSpawner,
                               BasicBlock *tfEntry) override;

  /// Pre-process the root function \p f as a function that can spawn subtasks.
  void preProcessRootSpawner(Function &f, BasicBlock *tfEntry) override;

  /// Post-process the root Function \p f as a function that can spawn subtasks.
  void postProcessRootSpawner(Function &f, BasicBlock *tfEntry) override;

  /// Process the invocation of a task for an outlined function. This routine
  /// is invoked after processSpawner once for each child subtask.
  void processSubTaskCall(TaskOutlineInfo &toi, DominatorTree &dt) override;

  /// Create a custom loop outline processor for this tapir target.
  virtual LoopOutlineProcessor *
  getLoopOutlineProcessor(const TapirLoopInfo *tl) override = 0;
};

/// \@}

} // namespace llvm

#endif // KITSUNE_TARGETS_CPUTT_COMMON_H
