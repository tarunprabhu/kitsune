//===- PthreadsTT.h - Tapir target using POSIX threads ---------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tapir target that lowers to Kitsune pthreads runtime. This runtime targets
// POSIX threads.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TAPIR_PTHREADS_TT_H
#define LLVM_TAPIR_PTHREADS_TT_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {

class TTOptions;

class PthreadsLoop : public LoopOutlineProcessor {
public:
  /// Create a loop outline processor for the pthreads tapir target.
  /// \param M The host module
  /// \param TTOpts The tapir target options
  PthreadsLoop(Module &m, const TTOptions &ttOpts);
  ~PthreadsLoop();

  /// Returns an ArgStructMode enum value describing how inputs to the
  /// underlying task of a tapir loop should be passed to the task.
  ArgStructMode getArgStructMode() const override final;

  /// Processes a call to an outlined helper function for a tapir loop \ref tl.
  void processOutlinedLoopCall(TapirLoopInfo &tl, TaskOutlineInfo &toi,
                               DominatorTree &dt) override final;
};

class PthreadsTT : public TapirTarget {
public:
  PthreadsTT(Module &m, const TTOptions &ttOpts);
  virtual ~PthreadsTT() = default;

  /// Lower a call to the tapir.loop.grainsize intrinsic into a grainsize
  /// (coarsening) value.
  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;

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
  void postProcessHelper(Function &F) override final {
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
  getLoopOutlineProcessor(const TapirLoopInfo *tli) override final;
};

} // namespace llvm

#endif // LLVM_TAPIR_PTHREADS_TT_H
