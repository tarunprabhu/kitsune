//===--- SemaKitsune.cpp - Semantic analysis for Kitsune extensions -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for Kitsune extensions.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtKitsune.h"
#include "clang/Sema/SemaInternal.h"

using namespace clang;
using namespace sema;

StmtResult
Sema::ActOnKitsuneSpawnStmt(SourceLocation SpawnLoc,
                            StringRef sv,
                            Stmt *SubStmt) {
  DiagnoseUnusedExprResult(SubStmt, diag::warn_unused_expr);

  PushFunctionScope();
  // TODO: Figure out how to prevent jumps into and out of the spawned
  // substatement.
  setFunctionHasBranchProtectedScope();
  PushExpressionEvaluationContext(
      ExpressionEvaluationContext::PotentiallyEvaluated);

  StmtResult Result = new (Context) KitsuneSpawnStmt(SpawnLoc, sv, SubStmt);

  PopExpressionEvaluationContext();
  PopFunctionScopeInfo();

  return Result;
}

StmtResult
Sema::ActOnKitsuneSyncStmt(SourceLocation SyncLoc, StringRef sv) {
  return new (Context) KitsuneSyncStmt(SyncLoc, sv);
}

