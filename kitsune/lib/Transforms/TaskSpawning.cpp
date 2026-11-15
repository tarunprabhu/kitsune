//===- TaskSpawning.cpp - Spawn tasks -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Spawn tasks. This is the equivalent of the loop-spawning pass, but for
// "tasks", i.e. Single Entry Single Exit regions demarcated by detach and
// reattach instructions that are not part of a tapir loop.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/TaskSpawning.h"
#include "kitsune/Core/InstAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "kit-task-spawning"

using namespace llvm;

namespace {

class TaskSpawning {
protected:
  DominatorTree &dt;
  LoopInfo &li;

protected:
  bool serialize(DetachInst *inst);
  bool serialize(ReattachInst *inst);
  bool serialize(Instruction *inst);

public:
  TaskSpawning(DominatorTree &dt, LoopInfo &li) : dt(dt), li(li) {}

  bool runOn(Function &f);
};

} // namespace

bool TaskSpawning::serialize(DetachInst *detach) {
  BasicBlock *bb = detach->getParent();
  BasicBlock *body = detach->getDetached();
  BasicBlock *cont = detach->getContinue();

  ReplaceInstWithInst(detach, UncondBrInst::Create(body));
  dt.deleteEdge(bb, cont);

#ifndef NDEBUG
  dt.verify();
#endif // NDEBUG

  return true;
}

bool TaskSpawning::serialize(ReattachInst *reattach) {
  BasicBlock *cont = reattach->getDetachContinue();
  ReplaceInstWithInst(reattach, UncondBrInst::Create(cont));

  return true;
}

bool TaskSpawning::serialize(Instruction *inst) {
  if (auto *detach = dyn_cast<DetachInst>(inst))
    return serialize(detach);
  else if (auto *reattach = dyn_cast<ReattachInst>(inst))
    return serialize(reattach);
  return false;
}

bool TaskSpawning::runOn(Function &f) {
  // TODO: We don't have any real support for tasks in Kitsune just yet. The
  // only reason for this pass is to deal with the case where a tapir loop has
  // a single iteration and gets optimized away leaving the detach and reattach
  // instructions outside a loop. This only happens with tapir loops with the
  // serial tapir target.
  //
  // We could try and pretend to find tasks here, but we might as well just
  // implement what this pass will currently work on.
  SmallVector<Instruction *> taskBounds;
  for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
    if (isa<DetachInst>(*i) || isa<ReattachInst>(*i))
      if (!getTapirLoopFor(*i, li))
        if (getTargetAttr(*i) == TTID::Serial)
          taskBounds.push_back(&*i);

  bool changed = false;
  for (Instruction *inst : taskBounds)
    changed |= serialize(inst);
  return changed;
}

PreservedAnalyses TaskSpawningPass::run(Module &m, ModuleAnalysisManager &am) {
  FunctionAnalysisManager &fam =
      am.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();

  bool changed = false;
  for (Function &f : m) {
    if (f.size()) {
      DominatorTree &dt = fam.getResult<DominatorTreeAnalysis>(f);
      LoopInfo &li = fam.getResult<LoopAnalysis>(f);

      changed |= TaskSpawning(dt, li).runOn(f);
    }
  }

  if (changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
