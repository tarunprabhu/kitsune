//===- ValueUtils.cpp - Utilities for LLVM Value's ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for LLVM values.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ValueUtils.h"
#include "kitsune/Core/ArgUtils.h"
#include "kitsune/Core/BasicBlockUtils.h"
#include "kitsune/Core/ConstantUtils.h"
#include "kitsune/Core/FuncUtils.h"
#include "kitsune/Core/GVUtils.h"
#include "kitsune/Core/InstUtils.h"
#include "llvm/IR/Constants.h"

using namespace llvm;

static const Constant *asConst(const Value *v) {
  if (const auto *c = dyn_cast<Constant>(v))
    return stripCasts(c);
  return nullptr;
}

bool llvm::isZero(const Value *v) {
  if (const Constant *c = asConst(v)) {
    if (const auto *cint = dyn_cast<ConstantInt>(c))
      return cint->isZero();
    else if (const auto *cfp = dyn_cast<ConstantFP>(c))
      return cfp->isZero();
  }
  return false;
}

bool llvm::isZero(const Value *v, Type *ty) {
  return v->getType() == ty && isZero(v);
}

bool llvm::isIntOne(const Value *v) {
  if (const Constant *c = asConst(v))
    if (const auto *cint = dyn_cast<ConstantInt>(c))
      return cint->isOne();
  return false;
}

bool llvm::isIntOne(const Value *v, Type *ty) {
  return v->getType() == ty && isIntOne(v);
}

template <typename M, typename T,
          std::enable_if_t<std::is_same_v<std::remove_cv_t<T>, Value>, int> = 0>
static M *getModuleImpl(T &v) {
  if (auto *a = dyn_cast<Argument>(&v))
    return getModule(*a);
  else if (auto *bb = dyn_cast<BasicBlock>(&v))
    return getModule(*bb);
  else if (auto *f = dyn_cast<Function>(&v))
    return getModule(*f);
  else if (auto *g = dyn_cast<GlobalVariable>(&v))
    return getModule(*g);
  else if (auto *inst = dyn_cast<Instruction>(&v))
    return getModule(*inst);
  return nullptr;
}

Module *llvm::getModule(Value &v) { return getModuleImpl<Module>(v); }

const Module *llvm::getModule(const Value &v) {
  return getModuleImpl<const Module>(v);
}

std::string llvm::getName(const Value &v) {
  if (v.hasName())
    return v.getName().str();

  if (auto *a = dyn_cast<Argument>(&v))
    return getName(*a);
  else if (auto *bb = dyn_cast<BasicBlock>(&v))
    return getName(*bb);
  else if (auto *f = dyn_cast<Function>(&v))
    return getName(*f);
  else if (auto *g = dyn_cast<GlobalVariable>(&v))
    return getName(*g);
  else if (auto *inst = dyn_cast<Instruction>(&v))
    return getName(*inst);

  std::string buf;
  raw_string_ostream os(buf);

  v.printAsOperand(os, /*PrintType=*/false, getModule(v));
  return buf;
}
