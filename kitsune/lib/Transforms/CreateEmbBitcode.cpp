//===- CreateEmbBitcode.cpp - Create an embedded bitcode global -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Create an embedded module to create embedded bitcode. Clone the device
// functions into it.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/CreateEmbBitcode.h"
#include "kitsune/Analysis/TapirTargetAnalysis.h"
#include "kitsune/Core/CloningUtils.h"
#include "kitsune/Core/EmbBitcodeUtils.h"
#include "kitsune/Core/ReachableGlobals.h"
#include "kitsune/Core/EmbDeviceCodeUtils.h"
#include "kitsune/Support/TTUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include <functional>

#define DEBUG_TYPE "kit-embs"

using namespace llvm;

using GetLoopInfo = std::function<LoopInfo &(Function &)>;
using GetTaskInfo = std::function<TaskInfo &(Function &)>;

static std::vector<TTID> getTTsGeneratingEmbBC(Module &m,
                                               const TapirTargetInfo &tgi) {
  // TapirTargetInfo::getRequiredTTs() will return an empty vector if there are
  // no tapir loops in the module. Even so, if the primary tapir target
  // generates embedded bitcode, and there is at least one device function in
  // the module, an embedded bitcode module will have to be created for it.
  TTID mainTT = tgi.getTTID();
  std::vector<TTID> tts;
  if (ttUsesEmbBC(mainTT))
    tts.push_back(mainTT);

  for (TTID tt : tgi.getRequiredTTs(m))
    if (ttUsesEmbBC(tt) and tt != mainTT)
      tts.push_back(tt);

  return tts;
}

static std::vector<Function *> getDeviceFuncs(Module &m) {
  std::vector<Function *> funcs;
  for (Function &f : m.functions())
    if (f.hasFnAttribute(Attribute::KitDevice))
      funcs.push_back(&f);
  return funcs;
}

static std::vector<Loop *> getTapirLoops(Module &m, GetLoopInfo getLoopInfo,
                                         GetTaskInfo getTaskInfo) {
  std::vector<Loop *> loops;
  for (Function &f : m.functions()) {
    if (f.size()) {
      LoopInfo &li = getLoopInfo(f);
      TaskInfo &ti = getTaskInfo(f);
      for (Loop *tl : li)
        for (Loop *loop : post_order(tl))
          if (getTaskIfTapirLoopStructure(loop, &ti))
            loops.push_back(loop);
    }
  }
  return loops;
}

PreservedAnalyses CreateEmbBitcodePass::run(Module &m,
                                            ModuleAnalysisManager &mam) {
  const TapirTargetInfo &tgi = mam.getResult<TapirTargetAnalysis>(m);
  if (not tgi.hasTTID())
    return PreservedAnalyses::all();

  auto &fam = mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();
  auto getLoopInfo = [&fam](Function &f) -> LoopInfo & {
    return fam.getResult<LoopAnalysis>(f);
  };
  auto getTaskInfo = [&fam](Function &f) -> TaskInfo & {
    return fam.getResult<TaskAnalysis>(f);
  };

  const TapirTargetOptions &tto = tgi.getOptions();

  // If there are no tapir targets for which to create an embedded bitcode
  // module, there is no need to do anything.
  std::vector<TTID> tts = getTTsGeneratingEmbBC(m, tgi);
  if (tts.empty())
    return PreservedAnalyses::all();

  // If there are no tapir loops in the module, nor any device functions, an
  // embedded module need not be generated.
  std::vector<Function *> deviceFuncs = getDeviceFuncs(m);
  std::vector<Loop *> loops = getTapirLoops(m, getLoopInfo, getTaskInfo);
  if (deviceFuncs.empty() and loops.empty())
    return PreservedAnalyses::all();

  // The device functions could use other functions and global variables in the
  // module. In that case, those should be cloned into the device module as
  // well.
  ReachableGlobals usedGlobals;
  for (Function *f : deviceFuncs)
    usedGlobals.analyze(*f);

  // Create device modules for the required targets and clone the device
  // functions into them. The bitcode module must be accompanied by a singleton
  // fat binary global declaration.
  for (TTID tt : tts) {
    std::unique_ptr<Module> devM = createEmbModule(tt, tto, m);
    cloneGlobalValuesInto(usedGlobals, tt, *devM);
    if (GlobalVariable *g = getEmbBCGlobal(tt, m))
      (void)resetEmbBCGlobal(*devM, *g);
    else
      (void)createEmbBCGlobal(*devM, tt, m);
    (void)createSingletonFBGlobal(tt, m);
  }

  // At best, this will add one or more global variables, so none of the
  // analyses should have been invalidated.
  return PreservedAnalyses::all();
}
