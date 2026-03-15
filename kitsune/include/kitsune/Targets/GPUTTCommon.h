//===- GPUTTCommon.h - Base class GPU-centric tapir targets ----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class for the 'cuda' and 'hip' tapir targets.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TARGETS_GPUTT_COMMON_H
#define KITSUNE_TARGETS_GPUTT_COMMON_H

#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {

class TTOptions;

/// \addtogroup kitsune
/// \@{

/// Base class used by the 'cuda' and 'hip' tapir targets. This implements some
/// common functionality shared by both targets and requires the targets to
/// implement others.
class GPUTTBase : public TapirTarget {
protected:
  /// The ID of the tapir target that is specializing this class.
  TTID tt;

  /// The host module that originally contained the tapir loops. This is the
  /// same as the "M" member of the TapirTarget class, but has a more readable
  /// name.
  Module &hostM;

  /// Currently, we create a single module into which all tapir loops are
  /// outlined. This will eventually be compiled to GPU machine code.
  Module devM;

  /// When outlining tapir loops into the device module, we need to generate a
  /// name for the outlined function. This name must be unique. In the absence
  /// of debug information, the computed outlined function name consists of a
  /// fixed base with an integer suffix that is incremented for each tapir loop
  /// that is encountered.
  unsigned nextKernelID = 0;

protected:
  GPUTTBase(TTID tt, Module &hostM, const TTOptions &tto);

  /// Construct the name to be used for the function into which the tapir loop
  /// \p tl will be outlined.
  std::string getNameForTapirLoop(const TapirLoopInfo &tl);

public:
  virtual ~GPUTTBase() = default;

  /// Lower a call to the tapir.loop.grainsize intrinsic into a grain size
  /// (coarsening) value.
  Value *lowerGrainsizeCall(CallInst *call) override;

  /// Lower the given Tapir sync instruction.
  void lowerSync(SyncInst &si) override;

  /// Process a host module before any lowering is performed. Unlike
  /// prepareModule(), this is called in by the loop-spawning pass.
  void preProcessModule() override;

  /// Process Function f before any function outlining is performed.  This
  /// routine should not modify the CFG structure.
  bool preProcessFunction(Function &f, TaskInfo &ti,
                          bool processingTapirLoops) override;

  /// Add attributes to the Function helper produced from outlining a task.
  void addHelperAttributes(Function &helper) override;

  /// Pre-process the Function f that has just been outlined from a task.
  void preProcessOutlinedTask(Function &f, Instruction *detachPt,
                              Instruction *tfCreate, bool isSpawner,
                              BasicBlock *bb) override;

  /// Post-process the Function f that has just been outlined from a task.
  void postProcessOutlinedTask(Function &f, Instruction *detachPtr,
                               Instruction *tfCreate, bool isSpawner,
                               BasicBlock *tfEntry) override;

  /// Pre-process the root Function f as a function that can spawn subtasks.
  void preProcessRootSpawner(Function &f, BasicBlock *tfEntry) override;

  /// Post-process the root Function f as a function that can spawn subtasks.
  void postProcessRootSpawner(Function &f, BasicBlock *tfEntry) override;

  /// Process the invocation of a task for an outlined function. This routine is
  /// invoked after processSpawner once for each child subtask.
  void processSubTaskCall(TaskOutlineInfo &toi, DominatorTree &dt) override;

  /// Process Function f at the end of the lowering process.
  void postProcessFunction(Function &f, bool outliningTapirLoops) override;

  /// Process the host-side module at the end of lowering all functions within
  /// the module.
  void postProcessModule() override;

  /// Process a generated helper Function f produced via outlining, at the end
  /// of the lowering process.
  void postProcessHelper(Function &f) override;

  /// Return a loop outline processor to process the given tapir loop. The
  /// returned object will be owned by the caller.
  LoopOutlineProcessor *
  getLoopOutlineProcessor(const TapirLoopInfo *tl) override = 0;
};

/// \@}

} // namespace llvm

#endif // KITSUNE_TARGETS_GPUTT_COMMON_H
