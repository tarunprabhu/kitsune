//===- ASTUtils.h - Utilities for clang's AST nodes -------------*- C++ -*-===//
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

#ifndef KITSUNE_CLANG_AST_UTILS_H
#define KITSUNE_CLANG_AST_UTILS_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"

namespace clang {

class ASTContext;
class Attr;
class Expr;
class Stmt;
class Type;

/// \addtogroup kitsune
/// @{

/// Strip implicit expressions and cleanups to retrieve the underlying
/// expression.
const Expr *getUnderlyingExpr(const Expr *expr);

/// Get the underlying unqualified desugared type of the expression.
const Type *getUnqualifiedDesugaredType(const Expr *expr);

/// Compute a name for the statement \stmt. \p attrs are the attributes attached
/// to the statement. If it contains a `kitsune::name` attribute, the value
/// specified there will be used. Otherwise, a name will be computed from the
/// location of statement.
std::string getNameFor(const Stmt &stmt, llvm::ArrayRef<const Attr *> attrs,
                       ASTContext &ast);

/// @}

} // namespace clang

#endif // KITSUNE_CLANG_AST_UTILS_H
