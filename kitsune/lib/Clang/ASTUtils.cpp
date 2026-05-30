//===- ASTUtils.cpp - Utilities for clang's AST nodes ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for clang's AST nodes.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Clang/ASTUtils.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"

using namespace clang;

const Expr *clang::getUnderlyingExpr(const Expr *expr) {
  const Expr *underlying = expr->IgnoreImplicit();
  if (const auto *ewc = dyn_cast<ExprWithCleanups>(underlying))
    return getUnderlyingExpr(ewc->getSubExpr());
  return underlying;
}

const clang::Type *clang::getUnqualifiedDesugaredType(const Expr *expr) {
  return expr->getType()->getUnqualifiedDesugaredType();
}
