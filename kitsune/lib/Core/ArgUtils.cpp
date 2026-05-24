//===- ArgUtils.cpp - Utilities for LLVM function arguments ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM function Argument's.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ArgUtils.h"
#include "llvm/IR/Function.h"

using namespace llvm;

Module *llvm::getModule(Argument &a) {
  if (Function *f = a.getParent())
    return f->getParent();
  return nullptr;
}

const Module *llvm::getModule(const Argument &a) {
  if (const Function *f = a.getParent())
    return f->getParent();
  return nullptr;
}

LLVMContext &llvm::getContext(const Argument &a) {
  return a.getParent()->getContext();
}

std::string llvm::getName(const Argument &a) {
  if (a.hasName())
    return a.getName().str();

  std::string buf;
  raw_string_ostream os(buf);
  a.printAsOperand(os, /*PrintType=*/false, getModule(a));

  return buf;
}
