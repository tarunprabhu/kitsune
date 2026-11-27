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

LLVMContext &llvm::getContext(const Argument &a) {
  return a.getParent()->getContext();
}

template <
    typename M, typename A,
    std::enable_if_t<std::is_same_v<std::remove_cv_t<A>, Argument>, int> = 0>
static M *getModuleImpl(A &a) {
  if (auto *f = a.getParent())
    return f->getParent();
  return nullptr;
}

Module *llvm::getModule(Argument &a) { return getModuleImpl<Module>(a); }

const Module *llvm::getModule(const Argument &a) {
  return getModuleImpl<const Module>(a);
}

Function *llvm::getFunction(Argument &a) { return a.getParent(); }

const Function *llvm::getFunction(const Argument &a) { return a.getParent(); }

std::string llvm::getName(const Argument &a) {
  if (a.hasName())
    return a.getName().str();

  std::string buf;
  raw_string_ostream os(buf);
  a.printAsOperand(os, /*PrintType=*/false, getModule(a));

  return buf;
}
