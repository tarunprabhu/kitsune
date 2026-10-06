//===- Serialize.cpp - Serialize certain tapir constructs -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass serializes certain tapir constructs.
//
// Currently, it only serializes certain tapir loops. These either cannot be
// lowered using a tapir target, or may degrade performance if lowered using a
// tapir target.
//
// In the future, it may also be used with standalone tapir tasks that are not
// currently supported.
//
// REQUIRES: kit-annotate-prelower
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/Serialize.h"
#include "kitsune/Analysis/TapirLoopNestAnalysis.h"
#include "kitsune/Core/Diagnostics.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/ModuleAttrs.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#define DEBUG_TYPE "kit-serialize"

using namespace llvm;

/// Verbose mode for this pass. It is probably not worth promoting this to a
/// frontend option. If this option is provided, a message will be emitted when
/// a loop is serialized.
///
/// FIXME: The current default is to emit remarks. This is not how LLVM
/// generally operates. Hever, as Kitsune is under heavy development and is
/// being used with more complex codes, it is useful for this pass to be verbose
/// by default.
static cl::opt<bool>
    clVerbose("serialize-verbose", cl::init(true), cl::Hidden,
              cl::cat(cl::catKitClDevOpts),
              cl::desc("The verbosity level of the kit-serialize "
                       "pass. Must be one of [0,1]"));

// If the given syncregion has only a single use, and the user is a sync
// instruction, remove both the sync instruction and the syncregion. Returns
// true if the syncregion was removed, false otherwise.
static void removeSyncRegionAndSync(Value *syncRegion) {
  if (syncRegion->hasOneUse()) {
    User *user = syncRegion->use_begin()->getUser();
    if (auto *syncInst = dyn_cast<SyncInst>(user)) {
      assert(syncInst->getNumSuccessors() == 1 &&
             "Sync instruction must have a single successor");
      BasicBlock *succ = syncInst->getSuccessor(0);
      BranchInst::Create(succ, syncInst->getIterator());
      syncInst->eraseFromParent();

      assert(isa<CallBase>(syncRegion) &&
             "syncregion in detach instruction must be a call");
      cast<CallBase>(syncRegion)->eraseFromParent();
    }
  }
}

// Reset the "perfect.depth" annotation on the root of the tapir loop nest
// to which the given loop belongs. Always returns true.
static bool resetMaxPerfectDepth(Loop &loop, unsigned newDepth) {
  Loop *curr = &loop;
  while (*getPerfectLevelAttr(*curr) > 1) {
    curr = curr->getParentLoop();
    assert(curr && "Perfectly nested tapir loop at level greater than one "
                   "must have a parent");
  }
  addPerfectDepthAttr(*curr, newDepth);
  return true;
}

// Serialize the loop depending on the loop level. Always returns true.
static bool serializeLoop(Loop &loop, Task &task) {
  unsigned perfectLevel = *getPerfectLevelAttr(loop);
  Value *syncRegion = task.getDetach()->getSyncRegion();

  if (clVerbose)
    emitDiagnostic(loop, DiagID::RemarkSerializedLoop);

  // TODO: Once there is better multi-target support, we may be able to just set
  // the tapir loop target to `serial` and let loop-spawning deal with it.
  serializeTapirLoop(loop, task);
  removeSyncRegionAndSync(syncRegion);

  if (perfectLevel > 3)
    // In this case, we must adjust the depth annotation at the root of the
    // loop nest. We start looking at the parent because loop itself will no
    // longer contain any tapir loop annotations. These are required by
    // resetMaxPerfectDepth().
    resetMaxPerfectDepth(*loop.getParentLoop(), 3);

  return true;
}

// We currently do not allow interleaving serial loops between two tapir
// loops. For instance, the following is not allowed:
//
//   forall (i ...)
//     for (j ...)
//       forall (k ...)
//
// We are, therefore, guaranteed to only have situations like the following
//
//   forall (i ...)            forall (i ...)          forall (i ...)
//     forall (j ...)            forall (j ...)          forall (j ...)
//       forall (k ...)            forall (k ...)
//         forall (l ...)
//
// On the GPU, we do not support tapir loops with a nesting level of greater
// than 3. Although this restriction is not strictly necessary on the CPU, we
// do the same for consistency, and also because it is not clear if there is
// any advantage to dealing with deeper nests in a special way.
//
// This function will return those tapir loops whose nesting level is greater
// than 3, regardless of their tapir target. The nesting level attribute is
// required to be present on all tapir loops at this point.
SetVector<Loop *> getLoopsToSerialize(LoopInfo &li) {
  auto shouldSerialize = [](const Loop &loop) -> bool {
    if (!isTapirLoop(loop))
      return false;

    assert(hasPerfectLevelAttr(loop) && "Loop has perfect.level attribute");
    if (*getPerfectLevelAttr(loop) <= 3)
      return false;

    return true;
  };

  SetVector<Loop *> loopsToSerialize;
  for (Loop *loop : li.getLoopsInPreorder())
    if (isTopLevelTapirLoop(*loop))
      for (Loop *subLoop : getAllSubLoops(*loop))
        if (shouldSerialize(*subLoop))
          loopsToSerialize.insert(subLoop);

  return loopsToSerialize;
}

static bool serializeImpl(Function &f, LoopInfo &li, TaskInfo &ti) {
  bool changed = false;
  for (Loop *loop : getLoopsToSerialize(li))
    changed |= serializeLoop(*loop, *getTaskIfTapirLoop(loop, &ti));
  return changed;
}

PreservedAnalyses SerializePass::run(Function &f, FunctionAnalysisManager &am) {
  LoopInfo &li = am.getResult<LoopAnalysis>(f);
  TaskInfo &ti = am.getResult<TaskAnalysis>(f);

  bool changed = serializeImpl(f, li, ti);
  if (changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
