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
#include "kitsune/Core/ValueUtils.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

Module *llvm::getModule(BasicBlock &bb) {
  if (Function *f = bb.getParent())
    return f->getParent();
  return nullptr;
}

const Module *llvm::getModule(const BasicBlock &bb) {
  if (const Function *f = bb.getParent())
    return f->getParent();
  return nullptr;
}

std::string llvm::getName(const BasicBlock &bb) {
  if (bb.hasName())
    return bb.getName().str();

  std::string buf;
  raw_string_ostream os(buf);

  bb.printAsOperand(os, /*PrintType=*/false, getModule(bb));
  return buf;
}

bool llvm::isDisconnected(const BasicBlock &bb) {
  return pred_size(&bb) == 0 && succ_size(&bb) == 0;
}

bool llvm::isOrphaned(const BasicBlock &bb) { return pred_size(&bb) == 0; }

bool llvm::isUnreachable(const BasicBlock &bb) {
  return bb.size() == 1 && isa<UnreachableInst>(bb.front());
}
