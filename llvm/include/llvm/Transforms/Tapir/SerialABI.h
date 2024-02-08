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

namespace llvm {

class SerialTTOptions;
class TapirTargetOptions;

class SerialABI : public TapirTarget {
public:
  SerialABI(Module &M, const TapirTargetOptions &opts);
  virtual ~SerialABI() = default;

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
