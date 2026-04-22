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

  /// There is nothing to synchronize since this tapir target will have
  /// serialized any tapir loops. This simply replaces the sync with an
  /// unconditional branch instruction.
  void lowerSync(SyncInst &si) override final;

  /// This will serialize all tapir loops with the serial tapir target in the
  /// function. Obviously, all the functionality of this tapir target is in this
  /// callback. The loop spawning pass will not invoke any other callbacks on
  /// tapir loops that are not outlined. As a result, if the loops are not
  /// serialized in this callback, detach and reattach instructions will remain
  /// in the loop after this tapir target has run. In the grand scheme of
  /// things, leaving the detaches and reattaches in will not cause any issues
  /// because the tapir-to-target pass will eventually remove them - effectively
  /// serializing the loop at that point. But we would like for this tapir
  /// target to actually serialize the loops. Besides, leaving them in breaks
  /// some tests that expect this target to have removed those instructions.
  bool preProcessFunction(Function &f, TaskInfo &ti,
                          bool processingTapirLoops) override final;

  void postProcessFunction(Function &f,
                           bool processingTapirLoops) override final {
    // Nothing to be done here
  }

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
};

} // namespace llvm

#endif // KITSUNE_TARGETS_SERIAL_TT_H
