//==- RecomputeKernelProperties.cpp - Update kernel properties global var --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Update the global variables containing the properties of functions launched
// in kernel launch calls.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Transforms/RecomputeKernelProperties.h"
#include "kitsune/Analysis/TapirTargetAnalysis.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/KernelProperties.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"

#include <optional>

#define DEBUG_TYPE "kit-kernel-properties"

using namespace llvm;

static TTID getTTID(CallBase &call) {
  auto *arg = cast<ConstantInt>(call.getArgOperand(0));
  if (std::optional<TTID> tt = createTTIDFrom(arg->getZExtValue()))
    return *tt;
  llvm_unreachable("Could not find TTID in kernel launch call");
}

static StringRef getKernelName(CallBase &call) {
  // The first string that is passed to the call will be the kernel name. It is
  // reasonable to expect that this will never change since there is little
  // reason to use a string to represent the tapir target id.
  for (Use &op : call.args())
    if (auto *g = dyn_cast<GlobalVariable>(&*op))
      if (g->hasInitializer())
        if (auto *cda = dyn_cast<ConstantDataArray>(g->getInitializer()))
          if (cda->isCString())
            return cda->getAsCString();
  llvm_unreachable("Could not find kernel name argument in kernel launch call");
}

static GlobalVariable *getKernelPropertiesGlobal(CallBase &call) {
  for (Use &op : call.args())
    if (auto *g = dyn_cast<GlobalVariable>(&*op))
      if (g->hasAttribute("kit_kernel_props"))
        return g;
  llvm_unreachable("Could not find kernel properties in kernel launch call");
}

namespace llvm {

PreservedAnalyses
RecomputeKernelPropertiesPass::run(Module &m, ModuleAnalysisManager &mam) {
  // If no primary tapir target has been set, the tapir target options will
  // not have been set, so there is nothing that we can do.
  const TapirTargetInfo &tgi = mam.getResult<TapirTargetAnalysis>(m);
  if (not tgi.hasTTID())
    return PreservedAnalyses::all();

  EmbModulesMapTy embMs = getEmbModules(m);
  Function *launchFn =
      Intrinsic::getOrInsertDeclaration(&m, Intrinsic::kitrt_launch_kernel);

  for (Use &u : launchFn->uses()) {
    if (auto *call = dyn_cast<CallBase>(u.getUser())) {
      if (call->getCalledFunction() == launchFn) {
        TTID tt = getTTID(*call);
        StringRef kname = getKernelName(*call);
        GlobalVariable *g = getKernelPropertiesGlobal(*call);

        assert(embMs.find(tt) != embMs.end() &&
               "Embedded module for tapir target not found");

        Function *kf = embMs.at(tt)->getFunction(kname);
        assert(kf && "Could not find kernel function being launched");

        ConstantStruct *c = getKernelPropertiesConstant(*kf);
        g->setInitializer(c);

        LLVM_DEBUG(dbgs() << "\tproperties:\n"
                          << "\t  " << kf->getName() << "\n"
                          << "\t    memory ops: " << c->getOperand(0) << "\n"
                          << "\t    fp ops:     " << c->getOperand(1) << "\n"
                          << "\t    int ops:    " << c->getOperand(2) << "\n"
                          << "\t    other ops:  " << c->getOperand(3) << "\n");
      }
    }
  }

  // At best, the initializers of one or more globals will have changed, but
  // nothing else, so all analyses on the module remain valid.
  return PreservedAnalyses::all();
}

} // namespace llvm
