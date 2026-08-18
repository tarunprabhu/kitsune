//===- SemaKitsune.cpp - Semantic analysis for Kitsune extensions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Semantic analysis for Kitsune-specific language constructs, attributes,
// builtins and other extensions.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaKitsune.h"
#include "kitsune/Clang/ASTUtils.h"
#include "kitsune/Core/KitOptions.h"
#include "kitsune/Core/Reductions.h"
#include "clang/AST/StmtKitsune.h"
#include "clang/Sema/Sema.h"

using namespace clang;
using namespace sema;

template <typename V> static constexpr Handled<V> notHandled;

StmtResult Sema::ActOnSpawnStmt(SourceLocation spawnLoc, StringRef sv,
                                Stmt *subStmt) {
  DiagnoseUnusedExprResult(subStmt, diag::warn_unused_expr);

  PushFunctionScope();
  // TODO: Figure out how to prevent jumps into and out of the spawned
  // substatement.
  setFunctionHasBranchProtectedScope();
  PushExpressionEvaluationContext(
      ExpressionEvaluationContext::PotentiallyEvaluated);

  StmtResult result = new (Context) SpawnStmt(spawnLoc, sv, subStmt);

  PopExpressionEvaluationContext();
  PopFunctionScopeInfo();

  return result;
}

StmtResult Sema::ActOnSyncStmt(SourceLocation syncLoc, StringRef sv) {
  return new (Context) SyncStmt(syncLoc, sv);
}

SemaKitsune::SemaKitsune(Sema &s) : SemaBase(s), sema(SemaRef) {}

bool SemaKitsune::checkMaxOneOccurrence(
    attr::Kind kind, const SmallVectorImpl<const Attr *> &attrs, int err) {
  const Attr *dup = nullptr;
  for (const Attr *attr : attrs) {
    if (attr->getKind() == kind) {
      if (dup) {
        Diag(attr->getLocation(), err) << attr;
        Diag(dup->getLocation(), diag::note_conflicting_attribute);
        return false;
      }
      dup = attr;
    }
  }
  return true;
}

bool SemaKitsune::handleMemAccessAttr(Decl *decl, const ParsedAttr &attr) {
  if (decl->isInvalidDecl())
    return true;

  // Check if there is only one access qualifier.
  if (auto *existingAttr = decl->getAttr<KitMemAccessAttr>()) {
    if (existingAttr->getSemanticSpelling() == attr.getSemanticSpelling()) {
      Diag(attr.getLoc(), diag::warn_duplicate_declspec)
          << attr.getAttrName()->getName() << attr.getRange();
    } else {
      Diag(attr.getLoc(), diag::err_kit_multiple_access_qualifiers)
          << decl->getSourceRange();
      decl->setInvalidDecl(true);
      return true;
    }
  }

  ASTContext &ctx = sema.Context;
  decl->addAttr(::new (ctx) KitMemAccessAttr(ctx, attr));

  return true;
}

Handled<bool> SemaKitsune::processDeclAttribute(Decl *decl,
                                                const ParsedAttr &attr) {
  switch (attr.getKind()) {
  case ParsedAttr::AT_KitMemAccess:
    return handleMemAccessAttr(decl, attr);
  default:
    return notHandled<bool>;
  }
}

Attr *SemaKitsune::handleTTAttr(Stmt *stmt, const ParsedAttr &attr,
                                SourceRange range) {
  StringRef str;
  SourceLocation argLoc;
  if (!sema.checkStringLiteralArgumentAttr(attr, 0, str, &argLoc)) {
    Diag(attr.getLoc(), diag::err_kit_tt_unknown);
    return nullptr;
  }

  TTAttr::TTAttrKind kind;
  if (!TTAttr::ConvertStrToTTAttrKind(str, kind)) {
    Diag(attr.getLoc(), diag::err_kit_tt_unknown);
    return nullptr;
  }

  if (kind == TTAttr::Custom) {
    Diag(attr.getLoc(), diag::err_kit_tt_custom);
    return nullptr;
  }

  ASTContext &ctx = sema.Context;
  Stmt::StmtClass clss = stmt->getStmtClass();

  // The attribute may only be added on certain statements. It is not
  // currently supported on spawn and sync statements.
  if (clss == Stmt::ForallStmtClass || clss == Stmt::CXXForallRangeStmtClass) {
    return ::new (ctx) TTAttr(ctx, attr, kind);
  } else if (auto *expr = dyn_cast<Expr>(stmt)) {
    if (sema.getKitOpts().getKokkos()) {
      // See if this is an attributed Kokkos statement (if so, there is a
      // CallExpr lurking further down in the AST). To find this CallExpr we
      // need to work past implicit expressions and any associated cleanups.
      //
      // The most important detail to note is that the attribute is really not
      // attached to the Kokkos statement, even though it might appear to be so
      // from the program syntax and thus the programmer's perspective.
      //
      // How lovely!
      //
      if (const auto *call = dyn_cast<CallExpr>(getUnderlyingExpr(expr))) {
        const FunctionDecl *decl = call->getDirectCallee();
        std::string name = decl->getQualifiedNameAsString();
        if (name == "Kokkos::parallel_for" || name == "Kokkos::parallel_reduce")
          return ::new (ctx) TTAttr(ctx, attr, kind);
      }
    }
  }

  Diag(attr.getLoc(), diag::err_kit_tt_attr_unsupported_stmt);
  return nullptr;
}

Attr *SemaKitsune::handleLaunchAttr(Stmt *stmt, const ParsedAttr &attr,
                                    SourceRange range) {
  ASTContext &ctx = getASTContext();

  Expr *expr = attr.getArgAsExpr(0);
  std::optional<llvm::APSInt> argVal = expr->getIntegerConstantExpr(ctx);
  if (!argVal.has_value()) {
    Diag(attr.getLoc(), diag::err_attribute_requires_positive_integer)
        << attr << /* positive (1 == non-negative) */ 0;
    return nullptr;
  }

  int val = argVal->getSExtValue();
  if (val <= 0) {
    Diag(attr.getLoc(), diag::err_attribute_requires_positive_integer)
        << attr << /* positive (1 == non-negative) */ 0;
    return nullptr;
  } else if (val > 1024) {
    Diag(attr.getLoc(), diag::err_kit_launch_range) << attr;
    return nullptr;
  }

  return ::new (ctx) KitLaunchAttr(ctx, attr, val);
}

Attr *SemaKitsune::handleStmtNameAttr(Stmt *stmt, const ParsedAttr &attr,
                                      SourceRange range) {
  StringRef str;
  SourceLocation argLoc;
  if (!sema.checkStringLiteralArgumentAttr(attr, 0, str, &argLoc))
    return nullptr;

  if (str.empty()) {
    Diag(attr.getLoc(), diag::err_kit_name_empty);
    return nullptr;
  }

  for (char c : str) {
    if (std::isspace(c)) {
      Diag(attr.getLoc(), diag::err_kit_name_space);
      return nullptr;
    } else if (!std::isprint(c)) {
      Diag(attr.getLoc(), diag::err_kit_name_printable);
      return nullptr;
    }
  }

  ASTContext &ctx = getASTContext();
  return ::new (ctx) KitStmtNameAttr(ctx, attr, str);
}

Handled<Attr *> SemaKitsune::processStmtAttribute(Stmt *stmt,
                                                  const ParsedAttr &attr,
                                                  SourceRange range) {
  switch (attr.getKind()) {
  case ParsedAttr::AT_KitLaunch:
    return handleLaunchAttr(stmt, attr, range);
  case ParsedAttr::AT_KitStmtName:
    return handleStmtNameAttr(stmt, attr, range);
  case ParsedAttr::AT_TT:
    return handleTTAttr(stmt, attr, range);
  default:
    return notHandled<Attr *>;
  }
}

void SemaKitsune::checkAttributes(const Stmt *stmt,
                                  const SmallVectorImpl<const Attr *> &attrs) {
  attr::Kind checkMaxOneAttrs[] = {attr::TT, attr::KitLaunch,
                                   attr::KitStmtName};
  for (attr::Kind kind : checkMaxOneAttrs)
    if (!checkMaxOneOccurrence(kind, attrs,
                               diag::err_kit_duplicate_attribute_stmt))
      return;
}

QualType SemaKitsune::handleMemAccessAttr(QualType type,
                                          const ParsedAttr &attr) {
  auto getExistingAttrName = [](const TypeDecl &typeDecl) -> StringRef {
    if (const auto *existingAttr = typeDecl.getAttr<KitMemAccessAttr>())
      return existingAttr->getSpelling();
    return "readwrite";
  };

  if (const auto *typedefTy = type->getAs<TypedefType>()) {
    const TypeDecl *typeDecl = typedefTy->getDecl();
    StringRef existingAttrName = getExistingAttrName(*typeDecl);
    StringRef attrName = attr.getAttrName()->getName();
    SourceLocation attrLoc = attr.getLoc();
    SourceRange attrRange = attr.getRange();

    if (existingAttrName == attrName)
      Diag(attrLoc, diag::warn_duplicate_declspec) << attrName << attrRange;
    else
      Diag(attrLoc, diag::err_kit_multiple_access_qualifiers);

    Diag(typeDecl->getBeginLoc(), diag::note_kit_typedef_access_qualifier)
        << existingAttrName;
  }
  return type;
}

QualType SemaKitsune::handleMobileAttr(QualType type, const ParsedAttr &attr) {
  if (type->isPointerType())
    return sema.Context.getMobilePointerType(type->getPointeeType());

  Diag(attr.getLoc(), diag::err_kit_mobile_on_non_pointer);
  return type;
}

Handled<QualType> SemaKitsune::processTypeAttribute(QualType type,
                                                    const ParsedAttr &attr) {
  switch (attr.getKind()) {
  case ParsedAttr::AT_KitMemAccess:
    return handleMemAccessAttr(type, attr);
  case ParsedAttr::AT_KitMobile:
    return handleMobileAttr(type, attr);
  default:
    return notHandled<QualType>;
  }
}

bool SemaKitsune::checkMobileFreeCall(CallExpr *theCall) {
  QualType argType = theCall->getArg(0)->getType();
  if (argType->isMobilePointerType())
    return true;

  Diag(theCall->getBeginLoc(), diag::err_kit_mobile_free_arg);
  return false;
}

bool SemaKitsune::checkMobileCastUnsafeCall(CallExpr *theCall) {
  QualType argType = theCall->getArg(0)->getType();
  if (argType->isPointerType() && !argType->isMobilePointerType())
    return true;

  Diag(theCall->getBeginLoc(), diag::err_kit_mobile_cast_arg);
  return false;
}

static bool checkReduceValueType(llvm::ReduceOp op, QualType argType) {
  const Type *type = argType->getUnqualifiedDesugaredType();
  switch (op) {
  case llvm::ReduceOp::Custom:
    llvm_unreachable("NOT YET IMPLEMENTED: checkReduceValue(ReduceOp::Custom)");
  case llvm::ReduceOp::Sum:
  case llvm::ReduceOp::Prod:
  case llvm::ReduceOp::Max:
  case llvm::ReduceOp::MaxLoc:
  case llvm::ReduceOp::Min:
  case llvm::ReduceOp::MinLoc:
    return !type->isBooleanType() && !type->isEnumeralType();
  case llvm::ReduceOp::LAnd:
  case llvm::ReduceOp::LOr:
  case llvm::ReduceOp::LXor:
    return type->isBooleanType();
  case llvm::ReduceOp::BAnd:
  case llvm::ReduceOp::BOr:
  case llvm::ReduceOp::BXor:
    return type->isIntegerType() && !type->isEnumeralType();
  }
  llvm_unreachable("checkReduceValueType: Reduction operator not handled");
}

bool SemaKitsune::checkReduceCall(CallExpr *theCall) {
  SourceLocation loc = theCall->getBeginLoc();
  const IntegerLiteral *iLit =
      dyn_cast<IntegerLiteral>(getUnderlyingExpr(theCall->getArg(1)));
  if (!iLit) {
    Diag(loc, diag::err_kit_reduce_op_not_literal);
    return false;
  }

  unsigned iOp = iLit->getValue().getLimitedValue();
  std::optional<llvm::ReduceOp> op = llvm::fromInt<llvm::ReduceOp>(iOp);
  if (!op.has_value()) {
    Diag(loc, diag::err_kit_reduce_op_unknown);
    return false;
  }

  if (*op == llvm::ReduceOp::Custom || *op == llvm::ReduceOp::MaxLoc ||
      *op == llvm::ReduceOp::MinLoc) {
    Diag(loc, diag::err_kit_reduce_op_nyi) << toString(*op);
    return false;
  }

  // Custom reducers might require additional arguments. We don't currently
  // support them, so require exactly three arguments.
  if (theCall->getNumArgs() != 3) {
    Diag(loc, diag::err_kit_reduce_num_args);
    return false;
  }

  const Expr *valueExpr = getUnderlyingExpr(theCall->getArg(2));
  const Type *valueType = valueExpr->getType()->getUnqualifiedDesugaredType();
  QualType valueTy(valueType, 0);
  const BuiltinType *builtinType = dyn_cast<BuiltinType>(valueType);
  if (!builtinType) {
    Diag(loc, diag::err_kit_reduce_type_unsupported) << valueTy;
    return false;
  }

  if (!builtinType->isInteger() && !builtinType->isFloatingPoint()) {
    Diag(loc, diag::err_kit_reduce_type_unsupported) << valueTy;
    return false;
  }

  const Expr *destExpr = getUnderlyingExpr(theCall->getArg(0));
  const Type *destType = destExpr->getType()->getUnqualifiedDesugaredType();
  const PointerType *ptrType = cast<PointerType>(destType);
  if (ptrType->getPointeeType()->getUnqualifiedDesugaredType() != valueType) {
    Diag(loc, diag::err_kit_reduce_type_mismatch) << valueTy;
    return false;
  }

  if (!checkReduceValueType(*op, valueTy)) {
    Diag(loc, diag::err_kit_reduce_op_type) << toString(*op) << valueTy;
    return false;
  }

  return true;
}

Handled<bool> SemaKitsune::checkBuiltinFunctionCall(unsigned builtin,
                                                    CallExpr *theCall) {
  switch (builtin) {
  case Builtin::BIkitsune_mobile_free:
    return checkMobileFreeCall(theCall);
  case Builtin::BI__kitsune_mobile_cast_unsafe:
    return checkMobileCastUnsafeCall(theCall);
  case Builtin::BI__kitsune_reduce:
    return checkReduceCall(theCall);
  default:
    break;
  }
  return notHandled<bool>;
}

bool SemaKitsune::checkMobileCast(Expr *srcExpr, QualType destType,
                                  bool allowStripMobile, SourceLocation loc,
                                  StringRef castKind) {
  QualType srcType = srcExpr->getType();
  if (!srcType->isMobilePointerType() && destType->isMobilePointerType()) {
    Diag(loc, diag::err_kit_cast_to_mobile);
    return false;
  }
  if (!allowStripMobile) {
    if (srcType->isMobilePointerType() && !destType->isMobilePointerType()) {
      Diag(loc, diag::err_kit_cast_away_mobile) << castKind;
      return false;
    }
  }
  return true;
}
