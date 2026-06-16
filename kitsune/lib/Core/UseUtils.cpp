//===- UseUtils.cpp - Utilities for LLVM's use objects --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM's Use objects.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/UseUtils.h"
#include "llvm/IR/Instruction.h"

using namespace llvm;

bool llvm::isUseInBlock(Use &use, BasicBlock &bb) {
  if (User *user = use.getUser())
    if (auto *inst = dyn_cast<Instruction>(user))
      if (inst->getParent() == &bb)
        return true;
  return false;
}

bool llvm::isUseInConstant(Use &use) {
  if (User *user = use.getUser())
    if (isa<Constant>(user))
      return true;
  return false;
}
