//===- SerialTT.h - Tapir target that serializes the loop ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tapir target that serializes the tapir loop.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_TARGETS_SERIAL_TT_H
#define KITSUNE_TARGETS_SERIAL_TT_H

#include "llvm/Transforms/Tapir/LoweringUtils.h"

namespace llvm {

class TTOptions;

/// Tapir target that serializes tapir loops.
/// \ingroup kitsune
class SerialTT : public TapirTarget {
public:
  SerialTT(Module &m, const TTOptions &ttOpts);
  virtual ~SerialTT() = default;

  /// This implementation always returns false since we do not want to outline
  /// when using this tapir target.
  bool shouldDoOutlining(const Function &f) const override final;

  /// Lower a call to the tapir.loop.grainsize intrinsic. In this case, this
  /// always returns 0 because this tapir target does not use a grainsize.
  Value *lowerGrainsizeCall(CallInst *call) override final;

  void lowerSync(SyncInst &si) override final;

  bool preProcessFunction(Function &f, TaskInfo &ti,
                          bool processingTapirLoops) override final {
    // This callback did not not modify the CFG. Therefore, return false.
    return false;
  }

  void postProcessFunction(Function &f,
                           bool processingTapirLoops) override final {
    // Nothing to be done here
  }

  /// Process a generated helper function \p f produced via outlining, at the
  /// end of the lowering process.
  void postProcessHelper(Function &f) override final {
    // This tapir target does not outline tapir loops
  }

  void preProcessOutlinedTask(Function &f, Instruction *detachPt,
                              Instruction *tfCreate, bool isSpawner,
                              BasicBlock *tfEntry) override final {
    // This tapir target does not outline tapir loops
  }

  void postProcessOutlinedTask(Function &f, Instruction *detachPt,
                               Instruction *tfCreate, bool isSpawner,
                               BasicBlock *tfEntry) override final {
    // This tapir target does not outline tapir loops
  }

  void preProcessRootSpawner(Function &f, BasicBlock *tfEntry) override final {
    // Nothing to do here because none of the functions processed by this tapir
    // target can spawn subtasks.
  }

  void postProcessRootSpawner(Function &f, BasicBlock *tfEntry) override final {
    // Nothing to do here because none of the functions processed by this tapir
    // target can spawn subtasks.
  }

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

#endif // KITSUNE_TARGETS_SERIAL_TT_H
