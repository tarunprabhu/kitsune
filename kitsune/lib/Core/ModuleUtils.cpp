//===- ModuleUtils.cpp - Utilities for LLVM modules -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of utilities for LLVM Module.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ModuleUtils.h"
#include "kitsune/Core/ModuleAttrs.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;

LLVMContext &llvm::getContext(const Module &m) { return m.getContext(); }

std::string llvm::getName(const Module &m) { return m.getName().str(); }

void llvm::addDeviceModuleFlagsAttr(Module &m, TTID tt) {
  addDeviceModuleFlagsAttr(m, tt, m.getName());
}

void llvm::cloneModuleFlagsMetadataInto(Module &devM, const Module &hostM) {
  // These are the module flags that should be cloned over. Others will be
  // ignored.
  SmallSet<StringRef, 8> flags = {"Debug Info Version", "Dwarf Version",
                                  "PIC Level", "PIE Level", "wchar_size"};

  NamedMDNode &nmd = *devM.getOrInsertModuleFlagsMetadata();
  if (const NamedMDNode *hostMD = hostM.getModuleFlagsMetadata())
    for (const MDNode *md : hostMD->operands())
      if (md->getNumOperands() > 1)
        if (auto *mdString = dyn_cast<MDString>(md->getOperand(1)))
          if (flags.contains(mdString->getString()))
            nmd.addOperand(MDNode::replaceWithPermanent(md->clone()));
}

void llvm::cloneIdentMetadataInto(Module &devM, const Module &hostM) {
  NamedMDNode &nmd = *devM.getOrInsertNamedMetadata("llvm.ident");
  if (const NamedMDNode *ident = hostM.getNamedMetadata("llvm.ident"))
    for (const MDNode *md : ident->operands())
      nmd.addOperand(MDNode::replaceWithPermanent(md->clone()));
}

Function *llvm::getOrInsertFunction(Module &m, StringRef name,
                                    FunctionType *fty) {
  return cast<Function>(m.getOrInsertFunction(name, fty).getCallee());
}
