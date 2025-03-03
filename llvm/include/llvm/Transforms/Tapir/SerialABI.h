//===- SerialABI.h - Replace Tapir with serial projection ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Serial back end, which is used to convert Tapir
// instructions into their serial projection.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TAPIR_SERIAL_ABI_H
#define LLVM_TAPIR_SERIAL_ABI_H

#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Tapir/TapirTargetOptions.h"

namespace llvm {

/// Options for the serial tapir target. Currently, there are none specific to
/// this tapir target, but it will inherit some common options from the parent.
class SerialABIOptions : public TapirTargetOptions {
public:
  explicit SerialABIOptions() : TapirTargetOptions(TTO_Serial) {}
  explicit SerialABIOptions(const SerialABIOptions &) = default;
  virtual ~SerialABIOptions() = default;

  SerialABIOptions &operator=(const SerialABIOptions &) = delete;

  virtual void readClOptions() override;
  virtual SerialABIOptions *clone() const override;

  static bool classof(const TapirTargetOptions *TTO) {
    return TTO->getKind() == TTO_Serial;
  }
};

class SerialABI : public TapirTarget {
public:
  SerialABI(Module &M, const SerialABIOptions &opts);
  virtual ~SerialABI() = default;

  const SerialABIOptions &getOptions() const override final;

  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
  void lowerSync(SyncInst &inst) override final;

  bool shouldDoOutlining(const Function &F) const override final {
    return false;
  }
  bool preProcessFunction(Function &F, TaskInfo &TI,
                          bool ProcessingTapirLoops) override final;
  void postProcessFunction(Function &F,
                           bool ProcessingTapirLoops) override final {}
  void postProcessHelper(Function &F) override final {}

  void preProcessOutlinedTask(Function &F, Instruction *DetachPt,
                              Instruction *TaskFrameCreate, bool IsSpawner,
                              BasicBlock *TFEntry) override final {}
  void postProcessOutlinedTask(Function &F, Instruction *DetachPt,
                               Instruction *TaskFrameCreate, bool IsSpawner,
                               BasicBlock *TFEntry) override final {}
  void preProcessRootSpawner(Function &F, BasicBlock *TFEntry) override final {}
  void postProcessRootSpawner(Function &F, BasicBlock *TFEntry) override final {
  }
  void processSubTaskCall(TaskOutlineInfo &TOI,
                          DominatorTree &DT) override final {}
};

} // namespace llvm

#endif // LLVM_TAPIR_SERIAL_ABI_H
