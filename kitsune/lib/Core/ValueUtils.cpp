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
#include "kitsune/Core/ConstantUtils.h"
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
