//===- FuncUtils.cpp - Utilities for LLVM functions -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM Function's.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/FuncUtils.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"

using namespace llvm;

Module *llvm::getModule(Function &f) { return f.getParent(); }
const Module *llvm::getModule(const Function &f) { return f.getParent(); }

LLVMContext &llvm::getContext(const Function &f) { return f.getContext(); }

std::string llvm::getName(const Function &f) {
  if (f.hasName())
    return f.getName().str();

  std::string buf;
  raw_string_ostream os(buf);
  f.printAsOperand(os, /*PrintType=*/false, f.getParent());

  return buf;
}

void llvm::copyAttrs(Function &dst, const Function &src) {
  for (Attribute attr : src.getAttributes().getFnAttrs())
    dst.addAttributeAtIndex(AttributeList::FunctionIndex, attr);

  for (Attribute attr : src.getAttributes().getRetAttrs())
    dst.addAttributeAtIndex(AttributeList::ReturnIndex, attr);

  dst.setCallingConv(src.getCallingConv());
  if (src.hasGC())
    dst.setGC(src.getGC());
  else
    dst.clearGC();
  if (src.hasPersonalityFn())
    dst.setPersonalityFn(src.getPersonalityFn());
  if (src.hasPrefixData())
    dst.setPrefixData(src.getPrefixData());
  if (src.hasPrologueData())
    dst.setPrologueData(src.getPrologueData());
}

void llvm::copyAttrs(Argument &dst, const Argument &src) {
  for (Attribute attr : src.getAttributes())
    dst.addAttr(attr);
}

bool llvm::sortBasicBlocks(Function &f) {
  if (!f.size())
    return false;

  SmallVector<BasicBlock *> bbs;
  for (BasicBlock *bb : ReversePostOrderTraversal<Function *>(&f))
    bbs.push_back(bb);

  for (unsigned i = 1, e = bbs.size(); i < e; ++i)
    bbs[i]->moveAfter(bbs[i - 1]);

  return true;
}
