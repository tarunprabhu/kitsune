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
#include "kitsune/Frontend/KitsuneOptions.h"
#include "clang/AST/StmtKitsune.h"
#include "clang/Sema/Sema.h"

using namespace clang;
using namespace sema;

template <typename V> static constexpr Handled<V> notHandled;

// TODO:? Is it worth making this available as an expression utility?
// Strip implicit expressions and cleanups to retrieve the underlying
// expression.
static Expr *getUnderlyingExpr(Expr *expr) {
  Expr *underlying = expr->IgnoreImplicit();
  if (auto *ewc = dyn_cast<ExprWithCleanups>(underlying))
    return getUnderlyingExpr(ewc->getSubExpr());
  return underlying;
}

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

StmtResult Sema::ActOnSyncStmt(SourceLocation SyncLoc, StringRef sv) {
  return new (Context) SyncStmt(SyncLoc, sv);
}

SemaKitsune::SemaKitsune(Sema &s) : SemaBase(s), sema(SemaRef) {}

bool SemaKitsune::handleMemAccessAttr(Decl *decl, const ParsedAttr &attr) {
  if (decl->isInvalidDecl())
    return true;

  // Check if there is only one access qualifier.
  if (auto *existingAttr = decl->getAttr<KitsuneMemAccessAttr>()) {
    if (existingAttr->getSemanticSpelling() == attr.getSemanticSpelling()) {
      Diag(attr.getLoc(), diag::warn_duplicate_declspec)
          << attr.getAttrName()->getName() << attr.getRange();
    } else {
      Diag(attr.getLoc(), diag::err_kitsune_multiple_access_qualifiers)
          << decl->getSourceRange();
      decl->setInvalidDecl(true);
      return true;
    }
  }

  ASTContext &ctx = sema.Context;
  decl->addAttr(::new (ctx) KitsuneMemAccessAttr(ctx, attr));

  return true;
}

Handled<bool> SemaKitsune::processDeclAttribute(Decl *decl,
                                                const ParsedAttr &attr) {
  switch (attr.getKind()) {
  case ParsedAttr::AT_KitsuneMemAccess:
    return handleMemAccessAttr(decl, attr);
  default:
    break;
  }
  return notHandled<bool>;
}

Attr *SemaKitsune::handleTapirTargetAttr(Stmt *stmt, const ParsedAttr &attr,
                                         SourceRange range) {
  StringRef str;
  SourceLocation argLoc;
  if (!sema.checkStringLiteralArgumentAttr(attr, 0, str, &argLoc)) {
    Diag(attr.getLoc(), diag::err_tapir_target_unknown);
    return nullptr;
  }

  TapirTargetAttr::TapirTargetAttrTy kind;
  if (!TapirTargetAttr::ConvertStrToTapirTargetAttrTy(str, kind)) {
    Diag(attr.getLoc(), diag::err_tapir_target_unknown) << str << argLoc;
    return nullptr;
  }

  ASTContext &ctx = sema.Context;
  Stmt::StmtClass clss = stmt->getStmtClass();
  // We only support a limited range of statements.  Make sure we are dealing
  // with one of them -- if not return an error.
  //
  // The attribute is not currently supported on spawn and sync statements.
  if (clss == Stmt::ForallStmtClass || clss == Stmt::CXXForallRangeStmtClass) {
    return ::new (ctx) TapirTargetAttr(ctx, attr, kind);
  } else if (auto *expr = dyn_cast<Expr>(stmt)) {
    if (sema.getKitsuneOpts().getKokkos()) {
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
          return ::new (ctx) TapirTargetAttr(ctx, attr, kind);
      }
    }
  }

  Diag(attr.getLoc(), diag::err_tapir_target_attr_unsupported_stmt);
  return nullptr;
}

Attr *SemaKitsune::handleTapirStrategyAttr(Stmt *stmt, const ParsedAttr &attr,
                                           SourceRange range) {
  bool errState = false;

  StringRef str;
  SourceLocation argLoc;
  if (!sema.checkStringLiteralArgumentAttr(attr, 0, str, &argLoc)) {
    Diag(attr.getLoc(), diag::err_tapir_strategy_unknown);
    errState = true;
  }

  TapirStrategyAttr::TapirStrategyTy kind;
  if (!TapirStrategyAttr::ConvertStrToTapirStrategyTy(str, kind)) {
    Diag(attr.getLoc(), diag::err_tapir_strategy_unknown) << str << argLoc;
    errState = true;
  }

  if (errState)
    return nullptr;
  return ::new (sema.Context) TapirStrategyAttr(sema.Context, attr, kind);
}

Attr *SemaKitsune::handleLaunchAttr(Stmt *stmt, const ParsedAttr &attr,
                                    SourceRange range) {
  ASTContext &ctx = getASTContext();

  Expr *expr = attr.getArgAsExpr(0);
  std::optional<llvm::APSInt> argVal = expr->getIntegerConstantExpr(ctx);
  if (!argVal.has_value()) {
    Diag(attr.getLoc(), diag::err_kitsune_launch_non_integral_type);
    return nullptr;
  }

  int val = argVal->getSExtValue();
  if (val <= 0) {
    Diag(attr.getLoc(), diag::err_attribute_requires_positive_integer)
        << attr << /* positive (1 == non-negative) */ 0;
    return nullptr;
  }

  unsigned tpb = static_cast<unsigned>(val);
  return ::new (ctx) KitsuneLaunchAttr(ctx, attr, tpb);
}

Handled<Attr *> SemaKitsune::processStmtAttribute(Stmt *stmt,
                                                  const ParsedAttr &attr,
                                                  SourceRange range) {
  switch (attr.getKind()) {
  case ParsedAttr::AT_TapirStrategy:
    return handleTapirStrategyAttr(stmt, attr, range);
  case ParsedAttr::AT_TapirTarget:
    return handleTapirTargetAttr(stmt, attr, range);
  case ParsedAttr::AT_KitsuneLaunch:
    return handleLaunchAttr(stmt, attr, range);
  default:
    break;
  }
  return notHandled<Attr *>;
}

QualType SemaKitsune::handleMemAccessAttr(QualType type,
                                          const ParsedAttr &attr) {
  auto getExistingAttrName = [](const TypeDecl &typeDecl) -> StringRef {
    if (const auto *existingAttr = typeDecl.getAttr<KitsuneMemAccessAttr>())
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
      Diag(attrLoc, diag::err_kitsune_multiple_access_qualifiers);

    Diag(typeDecl->getBeginLoc(), diag::note_kitsune_typedef_access_qualifier)
        << existingAttrName;
  }
  return type;
}

QualType SemaKitsune::handleMobileAttr(QualType type, const ParsedAttr &attr) {
  if (type->isPointerType())
    return sema.Context.getMobilePointerType(type->getPointeeType());

  Diag(attr.getLoc(), diag::err_kitsune_mobile_on_non_pointer);
  return type;
}

Handled<QualType> SemaKitsune::processTypeAttribute(QualType type,
                                                    const ParsedAttr &attr) {
  switch (attr.getKind()) {
  case ParsedAttr::AT_KitsuneMemAccess:
    return handleMemAccessAttr(type, attr);
  case ParsedAttr::AT_KitsuneMobile:
    return handleMobileAttr(type, attr);
  default:
    break;
  }
  return notHandled<QualType>;
}

bool SemaKitsune::checkMobileFreeCall(CallExpr *theCall) {
  QualType argType = theCall->getArg(0)->getType();
  if (argType->isMobilePointerType())
    return true;

  Diag(theCall->getBeginLoc(), diag::err_kitsune_mobile_free_arg);
  return false;
}

bool SemaKitsune::checkMobileCastUnsafeCall(CallExpr *theCall) {
  QualType argType = theCall->getArg(0)->getType();
  if (argType->isPointerType() && !argType->isMobilePointerType())
    return true;

  Diag(theCall->getBeginLoc(), diag::err_kitsune_mobile_cast_arg);
  return false;
}

Handled<bool> SemaKitsune::checkBuiltinFunctionCall(unsigned builtin,
                                                    CallExpr *theCall) {
  switch (builtin) {
  case Builtin::BIkitsune_mobile_free:
    return checkMobileFreeCall(theCall);
  case Builtin::BI__kitsune_mobile_cast_unsafe:
    return checkMobileCastUnsafeCall(theCall);
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
    Diag(loc, diag::err_kitsune_cast_to_mobile);
    return false;
  }
  if (!allowStripMobile) {
    if (srcType->isMobilePointerType() && !destType->isMobilePointerType()) {
      Diag(loc, diag::err_kitsune_cast_away_mobile) << castKind;
      return false;
    }
  }
  return true;
}
