//===- CPUTTCommon.cpp - Base class for CPU-centric tapir targets ---------===//
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

#include "CPUTTCommon.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

CPUTTBase::CPUTTBase(Module &m, const TTOptions &tto) : TapirTarget(m, tto) {}

bool CPUTTBase::shouldDoOutlining(const Function &f) const { return true; }

Value *CPUTTBase::lowerGrainsizeCall(CallInst *call) {
  // This is only called from the TapirToTarget pass. However, after loop
  // spawning, there will be nothing for that pass to do, so this is not
  // expected to be called. In case it is, fail catastrophically since it would
  // imply that something elsewhere has changed and this may have to be modified
  // to keep up.
  llvm_unreachable("CPUTTCommon: Unexpected invocation of lowerGrainsizeCall");
}

void CPUTTBase::lowerSync(SyncInst &si) {
  // This is only called from the TapirToTarget pass. However, after loop
  // spawning, there will be nothing for that pass to do, so this is not
  // expected to be called. In case it is, fail catastrophically since it would
  // imply that something elsewhere has changed and this may have to be modified
  // to keep up.
  llvm_unreachable("CPUTTCommon: Unexpected invocation of lowerSync");
}

bool CPUTTBase::preProcessFunction(Function &f, TaskInfo &ti,
                                   bool processingTapirLoops) {
  // Return false indicating that nothing was done by this callback.
  return false;
}

void CPUTTBase::postProcessFunction(Function &f, bool processingTapirLoops) {
  // Nothing to be done here
}

void CPUTTBase::postProcessHelper(Function &f) {
  // Nothing to be done here
}

void CPUTTBase::preProcessOutlinedTask(Function &f, Instruction *detachPt,
                                       Instruction *tfCreate, bool isSpawner,
                                       BasicBlock *tfEntry) {
  // Nothing to be done here
}

void CPUTTBase::postProcessOutlinedTask(Function &f, Instruction *detachPt,
                                        Instruction *tfCreate, bool isSpawner,
                                        BasicBlock *tfEntry) {
  // Nothing to be done here
}

void CPUTTBase::preProcessRootSpawner(Function &f, BasicBlock *tfEntry) {
  // Nothing processed by this tapir target can spawn subtasks.
}

void CPUTTBase::postProcessRootSpawner(Function &f, BasicBlock *tfEntry) {
  // Nothing processed by this tapir target can spawn subtasks.
}

void CPUTTBase::processSubTaskCall(TaskOutlineInfo &toi, DominatorTree &dt) {
  // Nothing handled by this tapir target spawns subtasks.
}
