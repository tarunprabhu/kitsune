//===- TTObjectsAnalysis.cpp - Analysis pass for tapir targets ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass analyzes the module and determines the tapir targets required by
// each function. The result of the pass is an object that owns the instances
// of the tapir target objects that are used to lower tapir constructs. It also
// owns the TTOptions object.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Analysis/TTObjectsAnalysis.h"
#include "kitsune/Core/LoopAttrs.h"
#include "kitsune/Support/CommandLineOptions.h"
#include "kitsune/Targets/TapirTargets.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#define DEBUG_TYPE "ttobjects"

using namespace llvm;

static cl::opt<bool>
    clDumpTTO("dump-tapir-target-options", cl::init(false),
              cl::desc("Dump the tapir target options if they have been set"),
              cl::Hidden, cl::cat(cl::catKitClOpts));

/// Empty vector of tapir targets to be used when
/// TTObjects::getRequiredTTs is called with a function that does not
/// contain any tapir loops.
static const std::vector<TTID> noTTs;

TTObjects::TTObjects(std::optional<TTOptions> ttOpts) : ttOpts(ttOpts) {}

void TTObjects::computeRequiredTTs(Module &m, GetLoopInfo getLoopInfo,
                                   GetTaskInfo getTaskInfo) {
  ttsInFunc.clear();
  ttsInModule.clear();

  SmallSet<TTID, 2> ttsForModule;
  for (Function &f : m.functions()) {
    if (not f.size())
      continue;

    LoopInfo &li = getLoopInfo(f);
    TaskInfo &ti = getTaskInfo(f);
    SmallSet<TTID, 2> ttsForFunc;
    for (const Loop *tl : li) {
      for (const Loop *loop : post_order(tl)) {
        if (getTaskIfTapirLoop(loop, &ti)) {
          TTID tt = *getTargetAttr(*loop);
          ttsForFunc.insert(tt);
          ttsForModule.insert(tt);
        }
      }
    }

    // Simply sort the tapir targets in ascending order so we have some
    // determinism. For multi-target mode, we won't rely on any particular order
    // in which these are returned.
    this->ttsInFunc[&f].assign(ttsForFunc.begin(), ttsForFunc.end());
    std::sort(this->ttsInFunc[&f].begin(), this->ttsInFunc[&f].end());
  }

  this->ttsInModule.assign(ttsForModule.begin(), ttsForModule.end());
  std::sort(this->ttsInModule.begin(), this->ttsInModule.end());
}

void TTObjects::addTT(TTID id, TapirTarget *tt) { tts[id] = tt; }

bool TTObjects::hasTT(TTID id) const { return tts.find(id) != tts.end(); }

TapirTarget *TTObjects::getTT(TTID id) const {
  assert(hasTT(id) && "TapirTarget has been created for ID");
  return tts.at(id);
}

TTID TTObjects::getTTID() const {
  assert(ttOpts && "Tapir target options have not been set");
  return ttOpts->getTTID();
}

std::optional<TTID> TTObjects::getTTIDOrNull() const {
  if (ttOpts)
    return ttOpts->getTTID();
  return std::nullopt;
}

const TTOptions &TTObjects::getOptions() const {
  assert(ttOpts && "Tapir target options have not been set");
  return *ttOpts;
}

ArrayRef<TTID> TTObjects::getRequiredTTs(Function &f) const {
  if (ttsInFunc.find(&f) == ttsInFunc.end())
    return noTTs;
  return ttsInFunc.at(&f);
}

ArrayRef<TTID> TTObjects::getRequiredTTs(Module &) const { return ttsInModule; }

bool TTObjects::invalidate(Module &, const PreservedAnalyses &pa,
                           ModuleAnalysisManager::Invalidator &) {
  // Just checking if the CFG is preserved should work.
  // If loop analyses are not preserved, then this analysis is invalid.
  auto lac = pa.getChecker<LoopAnalysis>();
  return not(lac.preserved() or lac.preservedSet<AllAnalysesOn<Function>>());
}

AnalysisKey TTObjectsAnalysis::Key;

TTObjectsAnalysis::TTObjectsAnalysis(std::optional<TTOptions> tto)
    : ttObjs(tto) {
  if (clDumpTTO and ttObjs.hasTTID())
    ttObjs.getOptions().print(outs(), /*all=*/true);
}

TTObjectsAnalysis::Result TTObjectsAnalysis::run(Module &m,
                                                 ModuleAnalysisManager &mam) {
  // If a primary tapir target has not been set, don't do anything more since
  // the lack of a primary tapir target implies that tapir lowering has not been
  // enabled.
  if (not ttObjs.hasTTID())
    return ttObjs;

  auto &fam = mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();
  auto getLoopInfo = [&](Function &f) -> LoopInfo & {
    return fam.getResult<LoopAnalysis>(f);
  };
  auto getTaskInfo = [&](Function &f) -> TaskInfo & {
    return fam.getResult<TaskAnalysis>(f);
  };

  ttObjs.computeRequiredTTs(m, getLoopInfo, getTaskInfo);

  const TTOptions &tto = ttObjs.getOptions();
  std::vector<TTID> ids = ttObjs.getRequiredTTs(m);
  ids.push_back(ttObjs.getTTID());
  for (TTID id : ids) {
    if (not ttObjs.hasTT(id)) {
      tts[id] = makeTT(id, m, tto);
      ttObjs.addTT(id, tts.at(id).get());
    }
  }

  return ttObjs;
}

char TTObjectsAnalysisWrapperPass::ID = 0;
INITIALIZE_PASS(TTObjectsAnalysisWrapperPass, DEBUG_TYPE,
                "Tapir Target Analysis", false, true)

TTObjectsAnalysisWrapperPass::TTObjectsAnalysisWrapperPass()
    : ImmutablePass(ID), ttObjs(std::nullopt) {
  initializeTTObjectsAnalysisWrapperPassPass(*PassRegistry::getPassRegistry());
}

TTObjectsAnalysisWrapperPass::TTObjectsAnalysisWrapperPass(
    std::optional<TTOptions> ttOpts)
    : ImmutablePass(ID), ttObjs(ttOpts) {
  initializeTTObjectsAnalysisWrapperPassPass(*PassRegistry::getPassRegistry());
}

void TTObjectsAnalysisWrapperPass::getAnalysisUsage(AnalysisUsage &au) const {
  au.setPreservesAll();
}

TTObjectsAnalysisWrapperPass::Result
TTObjectsAnalysisWrapperPass::getResult() const {
  return ttObjs;
}

ModulePass *
llvm::createTTObjectsAnalysisWrapperPass(std::optional<TTOptions> ttOpts) {
  return new TTObjectsAnalysisWrapperPass(ttOpts);
}
