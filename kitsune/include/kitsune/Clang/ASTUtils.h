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

namespace clang {

class Expr;
class Type;

/// \addtogroup kitsune
/// @{

/// Strip implicit expressions and cleanups to retrieve the underlying
/// expression.
const Expr *getUnderlyingExpr(const Expr *expr);

/// Get the underlying unqualified desugared type of the expression.
const Type *getUnqualifiedDesugaredType(const Expr *expr);

/// Compute a name from the given source location. The returned name is usually
/// of the form "<name>:<line>:<col>", but it need not be. The returned name
/// may also be an empty string.
std::string getNameFrom(FullSourceLoc loc);

/// @}

} // namespace clang

#endif // KITSUNE_CLANG_AST_UTILS_H
