//===- SerialTT.cpp - Tapir target that serializes tapir loops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tapir target that serializes tapir loops.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Targets/SerialTT.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/Tapir.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "serialtt"

SerialTT::SerialTT(Module &m, const TTOptions &tto) : TapirTarget(m, tto) {}

bool SerialTT::shouldDoOutlining(const Function &f) const { return false; }

Value *SerialTT::lowerGrainsizeCall(CallInst *call) {
  // In this tapir target, we do not use a grain size, so always return 0.
  Value *gs = ConstantInt::get(call->getType(), 0);
  call->replaceAllUsesWith(gs);
  return gs;
}

void SerialTT::lowerSync(SyncInst &si) {
  // This is only called from the TapirToTarget pass. If the sync instruction
  // has not already been removed (for instance, by the SimplifyCFG pass), we
  // can simply replace the instruction with an unconditional branch since there
  // is no parallel work here to sync.
  ReplaceInstWithInst(&si, BranchInst::Create(si.getSuccessor(0)));
}

bool SerialTT::preProcessFunction(Function &f, TaskInfo &ti,
                                  bool processingTapirLoops) {
  if (!processingTapirLoops)
    return false;

  // FIXME: Pass these analyses to the tapir target instead of doing this.
  bool changed = false;
  DominatorTree dt(f);
  LoopInfo li(dt);
  for (Loop *loop : li.getLoopsInPreorder())
    if (Task *task = getTaskIfTapirLoop(loop, &ti))
      if (getTargetAttr(*loop) == TTID::Serial) {
        SerializeDetach(task->getDetach(), task);
        changed |= true;
      }

  // Recompute the taskinfo analysis because it will be used by loop-spawning
  // after this returns. We need to recalculate the dominator tree because the
  // CFG will have changed when the loops are serialized.
  dt.recalculate(f);
  ti.recalculate(f, dt);

  return changed;
}
