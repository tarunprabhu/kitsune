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
#include "kitsune/Core/EmbBitcodeUtils.h"
#include "kitsune/Core/KernelProperties.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"

#define DEBUG_TYPE "kit-kernel-properties"

using namespace llvm;

namespace llvm {

PreservedAnalyses
RecomputeKernelPropertiesPass::run(Module &m, ModuleAnalysisManager &mam) {
  // If no primary tapir target has been set, the tapir target options will
  // not have been set, so there is nothing that we can do.
  const TapirTargetInfo &tgi = mam.getResult<TapirTargetAnalysis>(m);
  if (not tgi.hasTTID())
    return PreservedAnalyses::all();

  EmbModulesMapTy embMs = getEmbModules(m);

  for (GlobalVariable &g : m.globals()) {
    if (g.hasAttribute("kit_kernel_props")) {
      StringRef kname = g.getAttribute("kit_kernel_props").getValueAsString();
      TTID tt = g.getAttribute(Attribute::KitTT).getTTID();

      assert(embMs.find(tt) != embMs.end() &&
             "Embedded module for tapir target not found");

      Function *kf = embMs.at(tt)->getFunction(kname);
      assert(kf && "Could not find kernel function being launched");

      ConstantStruct *c = getKernelPropertiesConstant(*kf);
      g.setInitializer(c);

      LLVM_DEBUG(dbgs() << "\tproperties:\n"
                        << "\t  " << kname << "\n"
                        << "\t    memory ops: " << c->getOperand(0) << "\n"
                        << "\t    fp ops:     " << c->getOperand(1) << "\n"
                        << "\t    int ops:    " << c->getOperand(2) << "\n"
                        << "\t    other ops:  " << c->getOperand(3) << "\n");
    }
  }

  // At best, the initializers of one or more globals will have changed, but
  // nothing else, so all analyses on the module remain valid.
  return PreservedAnalyses::all();
}

} // namespace llvm
