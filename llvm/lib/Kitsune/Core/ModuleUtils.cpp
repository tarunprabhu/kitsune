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
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;

/// Name of a named metadata node Kitsune metadata for device modules.
static constexpr StringRef mdDeviceModuleFlags = "kitsune.device.module.flags";

bool llvm::hasDeviceModuleMetadata(const Module &m) {
  return m.getNamedMetadata(mdDeviceModuleFlags);
}

void llvm::addDeviceModuleMetadata(TTID tt, Module &m) {
  auto addOperandAt = [](NamedMDNode &nmd, unsigned i, MDNode *md) -> void {
    if (nmd.getNumOperands() > i)
      nmd.setOperand(i, md);
    else
      nmd.addOperand(md);
  };

  LLVMContext &ctx = m.getContext();
  Type *i32 = Type::getInt32Ty(ctx);
  NamedMDNode *nmd = m.getOrInsertNamedMetadata(mdDeviceModuleFlags);

  Constant *cTT = ConstantInt::get(i32, int(tt));

  addOperandAt(*nmd, 0, MDNode::get(ctx, ConstantAsMetadata::get(cTT)));
  addOperandAt(*nmd, 1, MDNode::get(ctx, MDString::get(ctx, m.getName())));
}

std::optional<TTID> llvm::getTTIDFromDeviceModuleMetadata(const Module &m) {
  if (const NamedMDNode *nmd = m.getNamedMetadata(mdDeviceModuleFlags))
    if (const MDNode *md = nmd->getOperand(0))
      if (const auto *cmd = dyn_cast<ConstantAsMetadata>(md->getOperand(0)))
        if (const auto *cint = dyn_cast<ConstantInt>(cmd->getValue()))
          return TTID(cint->getSExtValue());
  return std::nullopt;
}

std::optional<StringRef>
llvm::getNameFromDeviceModuleMetadata(const Module &m) {
  if (const NamedMDNode *nmd = m.getNamedMetadata(mdDeviceModuleFlags))
    if (const MDNode *md = nmd->getOperand(1))
      if (const auto *mds = dyn_cast<MDString>(md->getOperand(0)))
        return mds->getString();
  return std::nullopt;
}
