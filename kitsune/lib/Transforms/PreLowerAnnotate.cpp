//===- PreLowerAnnotate.cpp - Add annotations before tapir lowering -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass computes properties of tapir loops and adds them to the loop
// annotations. These will be read by passes that run later in the pipeline.
// These are intended to inform how the loop will be transformed prior to loop
// spawning and may also be used to affect how the loop will be spawned.
//
// For example, a tapir loop that performs a reduction will be annotated with
// the tapir.loop.reduction. If compiling for a GPU, a pass that runs before
// loop spawning will examine this annotation and transform the loop to a form
// suitable for computing parallel reductions on a GPU. That pass will ignore
// loops that do not contain the attribute.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/PreLowerAnnotate.h"
#include "kitsune/Analysis/TapirLoopNestAnalysis.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Core/LoopUtils.h"
#include "kitsune/Core/ModuleAttrs.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "kit-annotate-prelower"

using namespace llvm;

/// Annotate tapir loops for use by the GPU-centric tapir targets, 'cuda' and
/// 'hip'. Loop is the root of a tapir loop nest containing zero or more
/// perfectly nested tapir loops. This only adds the perfect depth and perfect
/// level annotations to the appropriate tapir loops and the lowering enabled
/// annotation to the root.
static void annotateTapirLoopsForGPU(Loop &root, ScalarEvolution &se) {
  std::unique_ptr<TapirLoopNest> nest = TapirLoopNest::create(root, se);
  assert(nest && "Loop must be a tapir loop");

  ArrayRef<Loop *> perfectLoops = nest->getPerfectTapirLoops();
  assert(perfectLoops.size() && "Root of tapir loop nest must be perfect");
  assert(perfectLoops[0] == &root &&
         "First perfect loop in tapir loop nest must be the root");

  // The "perfect.depth" annotation must only be set on the root of the
  // tapir loop nest. The "perfect.level" annotation must be added to all
  // loops, including the root.
  unsigned depth = nest->getMaxPerfectDepth();
  addLoweringEnabledAttr(root);
  addPerfectDepthAttr(root, depth);
  LLVM_DEBUG(dbgs() << "PreLowerAnnotate: Add lowering enable on root '"
                    << getName(root) << "'\n");
  LLVM_DEBUG(dbgs() << "PreLowerAnnotate: Add perfect depth '" << depth
                    << "' on root\n");

  for (unsigned d = 1; d <= depth; ++d) {
    Loop *loop = perfectLoops[d - 1];
    addPerfectLevelAttr(*loop, d);
    LLVM_DEBUG(dbgs() << "PreLowerAnnotate: Add perfect level on loop '"
                      << getName(*loop) << "'\n");
  }
}

PreservedAnalyses PreLowerAnnotatePass::run(Module &m,
                                            ModuleAnalysisManager &mam) {
  FunctionAnalysisManager &fam =
      mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();

  for (Function &f : m) {
    if (!f.size())
      continue;

    LoopInfo &li = fam.getResult<LoopAnalysis>(f);
    ScalarEvolution &se = fam.getResult<ScalarEvolutionAnalysis>(f);

    /// Find the subloops that are contained within a tapir loop nest consisting
    /// of loops that are to be run on a GPU. These will be ignored.
    SmallSet<Loop *, 8> ignore;
    for (Loop *loop : li.getLoopsInPreorder())
      if (isTopLevelTapirLoopForGPU(*loop))
        for (Loop *subLoop : getAllSubLoops(*loop))
          ignore.insert(subLoop);

    for (Loop *loop : li.getLoopsInPreorder()) {
      if (ignore.contains(loop))
        continue;

      if (isTopLevelTapirLoopForGPU(*loop))
        annotateTapirLoopsForGPU(*loop, se);
      else if (isTapirLoop(*loop))
        addLoweringEnabledAttr(*loop);
    }
  }

  // At best, this pass will only change the metadata on existing loops and the
  // module. It will not add or remove any loops, or change any other code.
  return PreservedAnalyses::all();
}
