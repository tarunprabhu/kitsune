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
#include "kitsune/Frontend/CommandLineOptions.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#define DEBUG_TYPE "kit-serialize"

using namespace llvm;

/// How verbose do we want this pass to be. For now, for our own sanity when
/// working with complex codes, this will always emit a remark when a loop is
/// serialized. But this verbosity can be controlled if needed. It is probably
/// not worth promoting this to a frontend option.
///
/// The valid values for this option are:
///
///   0  Disable all remarks and notes. This is how most other LLVM passes
///      operate by default.
///
///   1  Emit remarks, but not notes. These will only emit a message indicating
///      that a loop was serialized. If location information is available for
///      the loop (only if debug information is present), that will be emitted
///      in the remark.
///
///   2  In addition to the remark, print a serialized representation of the
///      loop that was serialized.
///
static cl::opt<unsigned> clSerializeVerbose(
    "serialize-verbose", cl::init(1U), cl::Hidden, cl::cat(cl::catKitClDevOpts),
    cl::desc("The verbosity level of the kit-serialize pass. Must be 0, 1, 2"));

/// Convert the loop to a string representation for diagnostics.
static SmallString<256> toString(const Loop &loop) {
  SmallString<256> buf;
  raw_svector_ostream os(buf);

  os << loop;
  return buf;
}

/// If the given syncregion has only a single use, and the user is a sync
/// instruction, remove both the sync instruction and the syncregion. Returns
/// true if the syncregion was removed, false otherwise.
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

/// Reset the "perfect.depth" annotation on the root of the tapir loop nest
/// to which the given loop belongs. Always returns true.
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

/// Serialize the loop depending on the loop level. Always returns true.
static bool serializeLoop(Loop &loop, Task &task) {
  if (clSerializeVerbose) {
    emitDiagnostic(loop, DiagID::RemarkSerializedLoop);
    if (clSerializeVerbose == 2)
      emitDiagnostic(DiagID::NoteSerializedLoop, toString(loop));
  }

  unsigned perfectLevel = getPerfectLevelAttr(loop).value_or(0);
  Value *syncRegion = task.getDetach()->getSyncRegion();

  // TODO: Once there is better multi-target support, rather than serializing
  // the loop here, we could just set the tapir loop target to `serial` on loops
  // that are to be serialized and let the loop-spawning pass deal with it.
  serializeTapirLoop(loop, task);
  removeSyncRegionAndSync(syncRegion);
  clearTapirLoopAttrs(loop);
  addSerializedAttr(loop);

  if (perfectLevel > 3)
    // In this case, we must adjust the depth annotation at the root of the
    // loop nest. We start looking at the parent because loop itself will no
    // longer contain any tapir loop annotations. These are required by
    // resetMaxPerfectDepth().
    resetMaxPerfectDepth(*loop.getParentLoop(), 3);

  return true;
}

/// Populate the output parameter \p loopsToSerialize with loops that have
/// one of the GPU-centric tapir targets, 'cuda' or 'hip', and any of the
/// following are true:
///
///   - It is perfectly nested within a tapir loop nest at a level greater
///     than 3.
///
///   - The loop is part of a tapir loop nest, but is not perfectly nested.
///
///   - The loop is perfectly nested within a loop nest, but at a level
///     greater than the depth of the outer tapir loop nest. This can be
///     because one of the ancestors of the tapir loop is a non-tapir loop.
///
static void populateGPULoopsToSerialize(LoopInfo &li,
                                        SetVector<Loop *> &loopsToSerialize) {
  auto shouldSerializeLoop = [](const Loop &loop) -> bool {
    if (!isTapirLoop(loop))
      return false;

    // In cases such as those shown below, the innermost forall loops will not
    // contain a perfect level annotation.
    //
    //   forall (i ...)
    //     for (j ...)
    //       forall (k ...)
    //
    // In such cases, getTapirLoopPerfectLevelAttr() will return 0.
    //
    // At the current time, multidimensional kernel launches can have at most 3
    // dimensions. If deeper perfectly nested tapir loops are found, serialize
    // them since there is not much else we can do.
    unsigned perfectLevel = getPerfectLevelAttr(loop).value_or(0);
    return perfectLevel == 0 || perfectLevel > 3;
  };

  for (Loop *loop : li.getLoopsInPreorder())
    if (isTopLevelTapirLoopForGPU(*loop))
      for (Loop *subLoop : getAllSubLoops(*loop))
        if (shouldSerializeLoop(*subLoop))
          loopsToSerialize.insert(subLoop);
}

SetVector<Loop *> getLoopsToSerialize(LoopInfo &li) {
  SetVector<Loop *> loopsToSerialize;
  populateGPULoopsToSerialize(li, loopsToSerialize);

  return loopsToSerialize;
}

/// Check the loops in the given function and serialize any that should be
/// serialized. Return true if at least one loop was serialized, false
/// otherwise.
static bool run(Function &f, FunctionAnalysisManager &am) {
  bool changed = false;
  LoopInfo &li = am.getResult<LoopAnalysis>(f);
  TaskInfo &ti = am.getResult<TaskAnalysis>(f);

  for (Loop *loop : getLoopsToSerialize(li))
    changed |= serializeLoop(*loop, *getTaskIfTapirLoop(loop, &ti));

  return changed;
}

PreservedAnalyses SerializePass::run(Module &m, ModuleAnalysisManager &mam) {
  bool changed = false;
  FunctionAnalysisManager &fam =
      mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();

  for (Function &f : m)
    if (f.size())
      changed |= ::run(f, fam);

  if (changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
