//===- SerializeTapirLoops.cpp - Serialize certain tapir loops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass to analyze tapir loop nests and serialize certain tapir loops.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/SerializeTapirLoops.h"
#include "kitsune/Analysis/TapirLoopNestAnalysis.h"
#include "kitsune/Core/CommandLineOptions.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#include <map>

#define DEBUG_TYPE "kit-serialize-tapir-loops"

using namespace llvm;

// We emit a "remark" every time a tapir loop is serialized. This is just for
// our own sanity because we want to know what Kitsune does, especially with
// more complex codes. But it can be turned off if needed. It's probably not
// worth promoting this to a top-level (i.e. frontend) option.
static cl::opt<bool> clSerializeQuiet(
    "serialize-quiet", cl::init(false), cl::Hidden,
    cl::desc("Do not emit remarks when serializing tapir loops"),
    cl::cat(cl::catKitClDevOpts));

// static std::string toString(StringRef file, unsigned line, unsigned col) {
//   std::string buf;
//   raw_string_ostream os(buf);

//   os << file << ":" << line << ":" << col;
//   os.flush();
//   return buf;
// }

static std::string getLoopLoc(const Loop &loop) {
  if (BasicBlock *latch = loop.getLoopLatch())
    if (DebugLoc dbg = latch->getTerminator()->getDebugLoc())
      if (const auto *scope = dyn_cast<DIScope>(dbg.getScope()))
        return llvm::join_items(":", scope->getFilename(),
                                std::to_string(dbg.getLine()),
                                std::to_string(dbg.getCol()));
  return "";
}

static void printRemark(StringRef msg, const Loop &loop) {
  bool hasColors = WithColor(errs()).colorsEnabled();
  std::string loc = getLoopLoc(loop);
  WithColor::remark();
  if (hasColors)
    errs().changeColor(raw_ostream::SAVEDCOLOR, /*bold=*/true);
  if (loc.size()) {
    errs() << loc << ": " << msg << "\n";
    if (hasColors)
      errs().resetColor();
  } else {
    Function *f = loop.getHeader()->getParent();
    errs() << msg << " in function '" << f->getName() << "'\n";
    if (hasColors)
      errs().resetColor();
    errs() << "    " << loop << "\n";
  }
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
  while (*getTapirLoopPerfectLevelAttr(*curr) > 1) {
    curr = curr->getParentLoop();
    assert(curr && "Perfectly nested tapir loop at level greater than one "
                   "must have a parent");
  }
  addTapirLoopPerfectDepthAttr(*curr, newDepth);
  return true;
}

/// Serialize the loop depending on the loop level. Return true if the loop
/// was serialized, false otherwise.
static void serializeLoop(Loop &loop, Task &task) {
  unsigned perfectLevel = getTapirLoopPerfectLevelAttr(loop).value_or(0);
  DetachInst *detach = task.getDetach();
  Value *syncRegion = detach->getSyncRegion();

  if (not clSerializeQuiet)
    printRemark("serialized tapir loop", loop);

  SerializeDetach(task.getDetach(), &task);
  removeSyncRegionAndSync(syncRegion);
  clearTapirLoopAttrs(loop);

  if (perfectLevel > 3)
    // In this case, we must adjust the depth annotation at the root of the
    // loop nest. We start looking at the parent because loop itself will no
    // longer contain any tapir loop annotations. These are required by
    // resetMaxPerfectDepth().
    resetMaxPerfectDepth(*loop.getParentLoop(), 3);
}

static bool shouldSerializeLoop(Loop &loop, TaskInfo &ti) {
  if (!isTapirLoop(loop, ti))
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
  unsigned perfectLevel = getTapirLoopPerfectLevelAttr(loop).value_or(0);
  return perfectLevel == 0 || perfectLevel > 3;
}

PreservedAnalyses SerializeTapirLoopsPass::run(Module &m,
                                               ModuleAnalysisManager &mam) {
  bool changed = false;
  FunctionAnalysisManager &fam =
      mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();

  for (Function &f : m) {
    if (!f.size())
      continue;

    LoopInfo &li = fam.getResult<LoopAnalysis>(f);
    TaskInfo &ti = fam.getResult<TaskAnalysis>(f);

    // Serialize certain tapir loops when the primary tapir target is one of the
    // GPU-centric tapir targets, 'cuda' and 'hip'. Currently, a tapir loop is
    // serialized if any of the following conditions hold:
    //
    //   - It is perfectly nested within a tapir loop nest at a level greater
    //     than 3.
    //
    //   - The loop is part of a tapir loop nest, but is not perfectly nested.
    //
    //   - The loop is perfectly nested within a loop nest, but at a level
    //     greater than the depth of the outer tapir loop nest. This can be
    //     because one of the ancestors of the tapir loop is a non-tapir loop.
    //
    std::map<Loop *, Task *> loopsToSerialize;
    for (Loop *loop : li.getLoopsInPreorder())
      if (isTopLevelTapirLoopForGPU(*loop, ti))
        for (Loop *subLoop : getAllSubLoops(*loop))
          if (shouldSerializeLoop(*subLoop, ti))
            loopsToSerialize[subLoop] = getTaskIfTapirLoop(subLoop, &ti);

    for (auto &[loop, task] : loopsToSerialize)
      serializeLoop(*loop, *task);

    changed |= loopsToSerialize.size();
  }

  if (changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
