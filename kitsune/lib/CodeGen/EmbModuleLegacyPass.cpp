//==- EmbModulePass.cpp - Embedded module pass for the legacy pass manager -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class for passes that operate on embedded modules for the legacy pass
// manager. These typically perform transformations on the embedded modules and
// update the global variables in the parent module that contain them.
//
//===----------------------------------------------------------------------===//

#include "kitsune/CodeGen/EmbModuleLegacyPass.h"
#include "kitsune/Core/EmbUtils.h"
#include "kitsune/Core/GVAttrs.h"
#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Support/ErrorHandling.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"

using namespace llvm;

EmbModuleLegacyPass::EmbModuleLegacyPass(char ID) : ModulePass(ID) {}

void EmbModuleLegacyPass::getAnalysisUsage(AnalysisUsage &au) const {}

bool EmbModuleLegacyPass::runOnModule(Module &m) {
  bool anyChanged = false;
  for (GlobalVariable *g : getGlobals(m, GVAttrKind::BitCode)) {
    Expected<std::unique_ptr<Module>> embMOrErr = parseEmbBCGlobal(*g);
    if (not embMOrErr)
      exitOnError(embMOrErr.takeError());
    std::unique_ptr<Module> embM = std::move(embMOrErr.get());

    TTID tt = *getBitCodeAttr(*g);
    bool thisChanged = this->runOnEmbModule(tt, *embM);
    if (thisChanged)
      resetEmbBCGlobal(*embM, *g);
    anyChanged |= thisChanged;
  }

  return anyChanged;
}
