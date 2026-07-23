//===- SemaKitsune.h - Semantic Analysis for Kitsune constructs -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Semantic analysis for Kitsune-specific attributes and builtins.
//
// Semantic analysis for Kitsune-specific language constructs may also be moved
// here in the future.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMAKITSUNE_H
#define LLVM_CLANG_SEMA_SEMAKITSUNE_H

#include "clang/AST/ASTFwd.h"
#include "clang/Sema/SemaBase.h"

#include <optional>

namespace clang {

class ParsedAttr;

/// Type indicating if a Kitsune construct was processed. If it was, returns
/// an object that evaluates to true. The object encapsulates a value that may
/// be used by the caller. This is just an alias for std::optional since the
/// latter does everything that we need.
template <typename V> using Handled = std::optional<V>;

class SemaKitsune : public SemaBase {
protected:
  Sema &sema;

protected:
  bool handleMemAccessAttr(Decl *decl, const ParsedAttr &attr);

  Attr *handleTTAttr(Stmt *stmt, const ParsedAttr &attr, SourceRange range);
  Attr *handleLaunchAttr(Stmt *stmt, const ParsedAttr &attr, SourceRange range);
  Attr *handleStmtNameAttr(Stmt *stmt, const ParsedAttr &attr,
                           SourceRange range);

  QualType handleMemAccessAttr(QualType type, const ParsedAttr &attr);
  QualType handleMobileAttr(QualType type, const ParsedAttr &attr);

  bool checkMobileFreeCall(CallExpr *theCall);
  bool checkMobileCastUnsafeCall(CallExpr *theCall);
  bool checkReduceCall(CallExpr *theCall);

  // Ensure that the an attribute with kind \p kind occurs at most once in the
  // list \p attrs. If it occurs more than once, emit the error \p err, and
  // return false. Otherwise, return true.
  bool checkMaxOneOccurrence(attr::Kind kind,
                             const SmallVectorImpl<const Attr *> &attrs,
                             int err);

public:
  SemaKitsune(Sema &sema);

  /// If the parsed attribute \p attr is Kitsune-specific, process it and return
  /// true - even if errors occurred during processing. If \p attr is not
  /// processed by this function, return `notHandled`.
  Handled<bool> processDeclAttribute(Decl *decl, const ParsedAttr &attr);

  /// If the parsed attribute \p attr is not Kitsune-specific, return
  /// `notHandled`.  Otherwise, return nullptr if any errors occurred when
  /// processing the attribute, or a semantic attribute object if no errors
  /// were occurred.
  Handled<Attr *> processStmtAttribute(Stmt *stmt, const ParsedAttr &attr,
                                       SourceRange range);

  /// If the parsed attribute \p attr is not Kitsune-specific, return
  /// `notHandled`. Otherwise, return nullptr if any errors occurred when
  /// processing the attribute. If no errors occurred, return `type` or a new
  /// type with the attribute applied - depending on \p attr.
  Handled<QualType> processTypeAttribute(QualType type, const ParsedAttr &attr);

  /// Check the semantic attributes for the statement.
  void checkAttributes(const Stmt *stmt,
                       const SmallVectorImpl<const Attr *> &attrs);

  /// If \p builtinID is not a Kitsune-specific builtin, return `notHandled`.
  /// Otherwise, check the call \p theCall and return false if any errors were
  /// found, or true if no errors occurred.
  Handled<bool> checkBuiltinFunctionCall(unsigned builtinID, CallExpr *theCall);

  /// If either the type of \p srcExpr, \p destType is a pointer with the
  /// `mobile` attribute, check if the cast from \p srcExpr to \p destType is
  /// valid. If it is not, return false. In all other cases, return true.
  ///
  /// \param srcExpr          The cast expression
  /// \param destType         The type \p srcExpr is being cast to
  /// \param allowStripMobile If true, casts that strip the mobile attribute
  ///                         are allowed
  /// \param loc              The source location
  /// \param castKind         A string describing the cast. This is only used in
  ///                         diagnostics. Examples include "static_cast",
  ///                         "C-style cast", "const_cast" etc.
  bool checkMobileCast(Expr *srcExpr, QualType destType, bool allowStripMobile,
                       SourceLocation loc, StringRef castKind);
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAKITSUNE_H
