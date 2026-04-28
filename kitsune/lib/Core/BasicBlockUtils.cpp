//===- BasicBlockUtils.cpp - Utilities for LLVM's Basic Blocks ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM's Basic Blocks.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/BasicBlockUtils.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

bool llvm::isDisconnected(const BasicBlock &bb) {
  return pred_size(&bb) == 0 && succ_size(&bb) == 0;
}

bool llvm::isOrphaned(const BasicBlock &bb) { return pred_size(&bb) == 0; }

bool llvm::isUnreachable(const BasicBlock &bb) {
  return bb.size() == 1 && isa<UnreachableInst>(bb.front());
}
