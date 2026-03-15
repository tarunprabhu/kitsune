//===- ConstantUtils.cpp - Helper functions for constants -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper functions for constants.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/ConstantUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

using namespace llvm;

GlobalVariable *llvm::createConstString(StringRef s, Module &m,
                                        StringRef name) {
  for (GlobalVariable &g : m.globals())
    if (g.isConstant() and g.hasInitializer())
      if (auto *cda = dyn_cast<ConstantDataArray>(g.getInitializer()))
        if (cda->isCString() and cda->getAsCString() == s)
          return &g;

  LLVMContext &ctx = m.getContext();
  Constant *init = ConstantDataArray::getString(ctx, s, true);
  Type *type = init->getType();
  auto *g = new GlobalVariable(m, type, /*IsConstant=*/true,
                               GlobalValue::PrivateLinkage, init, name);
  g->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  g->setAlignment(Align(1));

  return g;
}

Constant *llvm::stripCasts(Constant *c) {
  if (auto *cst = dyn_cast_or_null<ConstantExpr>(c))
    if (cst->isCast())
      return stripCasts(cst->getOperand(0));
  return c;
}

const Constant *llvm::stripCasts(const Constant *c) {
  if (const auto *cst = dyn_cast_or_null<ConstantExpr>(c))
    if (cst->isCast())
      return stripCasts((const Constant*)cst->getOperand(0));
  return c;
}
