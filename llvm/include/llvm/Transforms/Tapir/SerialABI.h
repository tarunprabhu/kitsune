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
#include "llvm/Transforms/Tapir/TapirTargetIDs.h"

namespace llvm {

/// Options for the serial tapir target. Currently, there are none.
class SerialABIOptions : public TapirTargetOptions {
public:
  SerialABIOptions() : TapirTargetOptions(TTO_Serial) {}
  virtual ~SerialABIOptions() = default;

  SerialABIOptions(const SerialABIOptions &) = delete;
  SerialABIOptions &operator=(const SerialABIOptions &) = delete;

  virtual SerialABIOptions *clone() const override {
    return new SerialABIOptions();
  }

  static bool classof(const TapirTargetOptions *TTO) {
    return TTO->getKind() == TTO_Serial;
  }
};

class SerialABI : public TapirTarget {
public:
  SerialABI(Module &M) : TapirTarget(M) {}
  ~SerialABI() {}

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
