//===- ParseKitsune.cpp - Parse Kitsune-specific constructs ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Parser interface for Kitsune-specific constructs
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Parse/RAIIObjectsForParser.h"

using namespace clang;

/// ParseSyncStatement
///       sync-statement:
///         'sync' identifier ';'
StmtResult Parser::ParseSyncStatement() {
  assert(Tok.is(tok::kw__kitsune_sync) && "Not a sync stmt!");
  SourceLocation SyncLoc = ConsumeToken(); // eat the '_kitsune_sync'

  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_unqualified_id) << 0;
    SkipUntil(tok::semi);
    return StmtError();
  }

  Token IdentTok = Tok;
  ConsumeToken();
  return Actions.ActOnSyncStmt(SyncLoc,
                               IdentTok.getIdentifierInfo()->getName());
}

/// ParseSpawnStatement
///       spawn-statement:
///         'spawn' identifier statement
StmtResult Parser::ParseSpawnStatement() {
  assert(Tok.is(tok::kw__kitsune_spawn) && "Not a spawn stmt!");
  SourceLocation SpawnLoc = ConsumeToken(); // eat the '_kitsune_spawn'.

  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_unqualified_id) << 0;
    return StmtError();
  }

  Token IdentTok = Tok;
  ConsumeToken();

  // Parse statement of spawned child
  StmtResult SubStmt = ParseStatement();
  if (SubStmt.isInvalid()) {
    SkipUntil(tok::semi);
    return StmtError();
  }
  return Actions.ActOnSpawnStmt(
      SpawnLoc, IdentTok.getIdentifierInfo()->getName(), SubStmt.get());
}

/// ParseForallStatement
///       _kitsune_forall-statement:
///         '_kitsune_forall' '(' expr ';' expr ';' expr ')' statement
///         '_kitsune_forall' '(' declaration expr ';' expr ';' expr ')'
///         statement
StmtResult Parser::ParseForallStatement(SourceLocation *TrailingElseLoc,
                                        LabelDecl *PrecedingLabel) {
  assert(Tok.is(tok::kw__kitsune_forall) && "Not a forall stmt!");
  SourceLocation ForLoc = ConsumeToken(); // eat the 'for'.

  SourceLocation CoawaitLoc;
  if (Tok.is(tok::kw_co_await))
    CoawaitLoc = ConsumeToken();

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "for";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXXorObjC = getLangOpts().C99 || getLangOpts().CPlusPlus ||
    getLangOpts().ObjC;

  // C99 6.8.5p5 - In C99, the for statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  //
  // C++ 6.4p3:
  // A name introduced by a declaration in a condition is in scope from its
  // point of declaration until the end of the substatements controlled by the
  // condition.
  // C++ 3.3.2p4:
  // Names declared in the for-init-statement, and in the condition of if,
  // while, for, and switch statements are local to the if, while, for, or
  // switch statement (including the controlled statement).
  // C++ 6.5.3p1:
  // Names declared in the for-init-statement are in the same declarative-region
  // as those declared in the condition.
  //
  // Always enter a ControlScope, even in C90 mode; this is harmless as it
  // doesn't cause declarations to bind to this scope. We use this to avoid
  // diagnosing a comma operator in e.g. the third part of a for loop when
  // '-Wcomma' is enabled.
  unsigned ScopeFlags = Scope::ControlScope |
                        (C99orCXXorObjC ? Scope::DeclScope : Scope::NoScope);
  ParseScope ForallScope(this, ScopeFlags);
  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  ExprResult Value;

  bool ForEach = false;
  StmtResult FirstPart;
  Sema::ConditionResult SecondPart;
  ExprResult Collection;
  ForRangeInfo ForRangeInfo;
  FullExprArg ThirdPart(Actions);
  ForRangeInfo.ExpansionStmt = nullptr;

  // RAII helper to enter a context if we're parsing an expansion statement.
  //
  // This is required because some parts of an expansion statement (e.g. the
  // init-statement) are not in a dependent context and must thus be parsed in
  // the parent context.
  struct [[nodiscard]] ExpansionStmtContextRAII : Sema::ContextRAII {
    ExpansionStmtContextRAII(Sema &S, struct ForRangeInfo &Info,
                             DeclContext *Ctx)
        : ContextRAII(S, Info.ExpansionStmt ? Ctx : S.CurContext,
                      /*NewThis=*/false) {}
  };

  assert(!Actions.CurContext->isExpansionStmt());
  if (Tok.is(tok::code_completion)) {
    cutOffParsing();
    Actions.CodeCompletion().CodeCompleteOrdinaryName(
        getCurScope(), C99orCXXorObjC ? SemaCodeCompletion::PCC_ForInit
                                      : SemaCodeCompletion::PCC_Expression);
    return StmtError();
  }

  ParsedAttributes attrs(AttrFactory);
  MaybeParseCXX11Attributes(attrs);

  SourceLocation EmptyInitStmtSemiLoc;

  // Parse the first part of the for specifier.
  if (Tok.is(tok::semi)) { // for (;
    ProhibitAttributes(attrs);
    // no first part, eat the ';'.
    SourceLocation SemiLoc = Tok.getLocation();
    if (!Tok.hasLeadingEmptyMacro() && !SemiLoc.isMacroID())
      EmptyInitStmtSemiLoc = SemiLoc;
    ConsumeToken();
  } else if (getLangOpts().CPlusPlus && Tok.is(tok::identifier) &&
             isForRangeIdentifier()) {
    // Note: This path is solely for error recovery if a user omits the type-id
    // and writes 'for (x : ...)'; normally, the for-range-declaration is parsed
    // in the 'if (isForInitDeclaration())' branch below.
    ProhibitAttributes(attrs);
    IdentifierInfo *Name = Tok.getIdentifierInfo();
    SourceLocation Loc = ConsumeToken();
    MaybeParseCXX11Attributes(attrs);

    ForRangeInfo.ColonLoc = ConsumeToken();
    ParseForRangeInitializerAfterColon(ForRangeInfo, /*VarDeclSpec=*/nullptr);

    Diag(Loc, diag::err_for_range_identifier)
        << (ForRangeInfo.ExpansionStmt != nullptr)
        << ((getLangOpts().CPlusPlus11 && !getLangOpts().CPlusPlus17)
                ? FixItHint::CreateInsertion(Loc, "auto &&")
                : FixItHint());

    if (!ForRangeInfo.ExpansionStmt)
      ForRangeInfo.LoopVar =
          Actions.ActOnCXXForRangeIdentifier(getCurScope(), Loc, Name, attrs);
  } else if (isForInitDeclaration()) { // for (int X = 4;
    ParenBraceBracketBalancer BalancerRAIIObj(*this);
    ExpansionStmtContextRAII EnterParentContext{
        Actions, ForRangeInfo, Actions.CurContext->getParent()};

    // Parse declaration, which eats the ';'.
    if (!C99orCXXorObjC) { // Use of C99-style for loops in C90 mode?
      Diag(Tok, diag::ext_c99_variable_decl_in_for_loop);
      Diag(Tok, diag::warn_gcc_variable_decl_in_for_loop);
    }
    DeclGroupPtrTy DG;
    SourceLocation DeclStart = Tok.getLocation(), DeclEnd;
    if (!getLangOpts().CPlusPlus &&
        Tok.isOneOf(tok::kw_static_assert, tok::kw__Static_assert)) {
      ProhibitAttributes(attrs);
      Decl *D = ParseStaticAssertDeclaration(DeclEnd);
      DG = Actions.ConvertDeclToDeclGroup(D);
      FirstPart = Actions.ActOnDeclStmt(DG, DeclStart, Tok.getLocation());
    } else if (Tok.is(tok::kw_using)) {
      DG = ParseAliasDeclarationInInitStatement(DeclaratorContext::ForInit,
                                                attrs);
      FirstPart = Actions.ActOnDeclStmt(DG, DeclStart, Tok.getLocation());
    } else {
      // In C++0x, "for (T NS:a" might not be a typo for ::
      bool MightBeForRangeStmt = getLangOpts().CPlusPlus || getLangOpts().ObjC;
      ColonProtectionRAIIObject ColonProtection(*this, MightBeForRangeStmt);
      ParsedAttributes DeclSpecAttrs(AttrFactory);
      DG = ParseSimpleDeclaration(
          DeclaratorContext::ForInit, DeclEnd, attrs, DeclSpecAttrs, false,
          MightBeForRangeStmt ? &ForRangeInfo : nullptr);
      FirstPart = Actions.ActOnDeclStmt(DG, DeclStart, Tok.getLocation());
      if (ForRangeInfo.ParsedForRangeDecl()) {
        Diag(ForRangeInfo.ColonLoc, getLangOpts().CPlusPlus11
                                        ? diag::warn_cxx98_compat_for_range
                                        : diag::ext_for_range);
        ForRangeInfo.LoopVar = FirstPart;
        FirstPart = StmtResult();
      } else if (Tok.is(tok::semi)) { // for (int x = 4;
        ConsumeToken();
      } else if ((ForEach = isTokIdentifier_in())) {
        Actions.ActOnForEachDeclStmt(DG);
        // ObjC: for (id x in expr)
        ConsumeToken(); // consume 'in'

        if (Tok.is(tok::code_completion)) {
          cutOffParsing();
          Actions.CodeCompletion().CodeCompleteObjCForCollection(getCurScope(),
                                                                 DG);
          return StmtError();
        }
        Collection = ParseExpression();
      } else {
        Diag(Tok, diag::err_expected_semi_for);
      }
    }
  } else {
    // An expression here should not be inside the expansion statement context.
    ExpansionStmtContextRAII EnterParentContext{
        Actions, ForRangeInfo, Actions.CurContext->getParent()};
    ProhibitAttributes(attrs);
    Value = ParseExpression();

    ForEach = isTokIdentifier_in();

    // Turn the expression into a stmt.
    if (!Value.isInvalid()) {
      if (ForEach)
        FirstPart = Actions.ActOnForEachLValueExpr(Value.get());
      else {
        // We already know this is not an init-statement within a for loop, so
        // if we are parsing a C++11 range-based for loop, we should treat this
        // expression statement as being a discarded value expression because
        // we will err below. This way we do not warn on an unused expression
        // that was an error in the first place, like with: for (expr : expr);
        bool IsRangeBasedFor =
            getLangOpts().CPlusPlus11 && !ForEach && Tok.is(tok::colon);
        FirstPart = Actions.ActOnExprStmt(Value, !IsRangeBasedFor);
      }
    }

    if (Tok.is(tok::semi)) {
      ConsumeToken();
    } else if (ForEach) {
      ConsumeToken(); // consume 'in'

      if (Tok.is(tok::code_completion)) {
        cutOffParsing();
        Actions.CodeCompletion().CodeCompleteObjCForCollection(getCurScope(),
                                                               nullptr);
        return StmtError();
      }
      Collection = ParseExpression();
    } else if (getLangOpts().CPlusPlus11 && Tok.is(tok::colon) &&
               FirstPart.get()) {
      // User tried to write the reasonable, but ill-formed, for-range-statement
      //   for (expr : expr) { ... }
      Diag(Tok, diag::err_for_range_expected_decl)
          << false << FirstPart.get()->getSourceRange();
      SkipUntil(tok::r_paren, StopBeforeMatch);
      SecondPart = Sema::ConditionError();
    } else {
      if (!Value.isInvalid()) {
        Diag(Tok, diag::err_expected_semi_for);
      } else {
        // Skip until semicolon or rparen, don't consume it.
        SkipUntil(tok::r_paren, StopAtSemi | StopBeforeMatch);
        if (Tok.is(tok::semi))
          ConsumeToken();
      }
    }
  }

  // Parse the second part of the for specifier.
  if (!ForEach && !ForRangeInfo.ParsedForRangeDecl() &&
      !SecondPart.isInvalid()) {
    // Parse the second part of the for specifier.
    if (Tok.is(tok::semi)) { // for (...;;
      // no second part.
    } else if (Tok.is(tok::r_paren)) {
      // missing both semicolons.
    } else {
      if (getLangOpts().CPlusPlus) {
        // C++2a: We've parsed an init-statement; we might have a
        // for-range-declaration next.
        bool MightBeForRangeStmt = !ForRangeInfo.ParsedForRangeDecl();
        ColonProtectionRAIIObject ColonProtection(*this, MightBeForRangeStmt);
        SourceLocation SecondPartStart = Tok.getLocation();
        Sema::ConditionKind CK = Sema::ConditionKind::Boolean;
        SecondPart = ParseCXXCondition(
            /*InitStmt=*/nullptr, ForLoc, CK,
            // FIXME: recovery if we don't see another semi!
            /*MissingOK=*/true, MightBeForRangeStmt ? &ForRangeInfo : nullptr);

        if (ForRangeInfo.ParsedForRangeDecl()) {
          Diag(FirstPart.get() ? FirstPart.get()->getBeginLoc()
                               : ForRangeInfo.ColonLoc,
               getLangOpts().CPlusPlus20
                   ? diag::warn_cxx17_compat_for_range_init_stmt
                   : diag::ext_for_range_init_stmt)
              << (FirstPart.get() ? FirstPart.get()->getSourceRange()
                                  : SourceRange());
          if (EmptyInitStmtSemiLoc.isValid()) {
            Diag(EmptyInitStmtSemiLoc, diag::warn_empty_init_statement)
                << /*for-loop*/ 2
                << FixItHint::CreateRemoval(EmptyInitStmtSemiLoc);
          }
        }

        if (SecondPart.isInvalid()) {
          ExprResult CondExpr = Actions.CreateRecoveryExpr(
              SecondPartStart,
              Tok.getLocation() == SecondPartStart ? SecondPartStart
                                                   : PrevTokLocation,
              {}, Actions.PreferredConditionType(CK));
          if (!CondExpr.isInvalid())
            SecondPart = Actions.ActOnCondition(getCurScope(), ForLoc,
                                                CondExpr.get(), CK,
                                                /*MissingOK=*/false);
        }

      } else {
        ExprResult SecondExpr = ParseExpression();
        if (SecondExpr.isInvalid())
          SecondPart = Sema::ConditionError();
        else
          SecondPart = Actions.ActOnCondition(
              getCurScope(), ForLoc, SecondExpr.get(),
              Sema::ConditionKind::Boolean, /*MissingOK=*/true);
      }
    }
  }

  // Parse the third part of the for statement.
  if (!ForEach && !ForRangeInfo.ParsedForRangeDecl()) {
    if (Tok.isNot(tok::semi)) {
      if (!SecondPart.isInvalid())
        Diag(Tok, diag::err_expected_semi_for);
      SkipUntil(tok::r_paren, StopAtSemi | StopBeforeMatch);
    }

    if (Tok.is(tok::semi)) {
      ConsumeToken();
    }

    if (Tok.isNot(tok::r_paren)) { // for (...;...;)
      ExprResult Third = ParseExpression();
      // FIXME: The C++11 standard doesn't actually say that this is a
      // discarded-value expression, but it clearly should be.
      ThirdPart = Actions.MakeFullDiscardedValueExpr(Third.get());
    }
  }
  // Match the ')'.
  T.consumeClose();

  // C++ Coroutines [stmt.iter]:
  //   'co_await' can only be used for a range-based for statement.
  if (CoawaitLoc.isValid() && !ForRangeInfo.ParsedForRangeDecl()) {
    Diag(CoawaitLoc, diag::err_for_co_await_not_range_for);
    CoawaitLoc = SourceLocation();
  }

  if (CoawaitLoc.isValid() && getLangOpts().CPlusPlus20)
    Diag(CoawaitLoc, diag::warn_deprecated_for_co_await);

  // We need to perform most of the semantic analysis for a C++0x for-range
  // statement before parsing the body, in order to be able to deduce the type
  // of an auto-typed loop variable.
  StmtResult ForallRangeStmt;
  StmtResult ForEachStmt;

  if (ForRangeInfo.ParsedForRangeDecl()) {
    ForallRangeStmt = Actions.ActOnCXXForallRangeStmt(
        getCurScope(), ForLoc, CoawaitLoc, FirstPart.get(),
        ForRangeInfo.LoopVar.get(), ForRangeInfo.ColonLoc,
        ForRangeInfo.RangeExpr.get(), T.getCloseLocation(), Sema::BFRK_Build);
  } else if (ForEach) {
    // Similarly, we need to do the semantic analysis for a for-range
    // statement immediately in order to close over temporaries correctly.
    ForEachStmt = Actions.ObjC().ActOnObjCForCollectionStmt(
        ForLoc, FirstPart.get(), Collection.get(), T.getCloseLocation());
  } else {
    // In OpenMP loop region loop control variable must be captured and be
    // private. Perform analysis of first part (if any).
    if (getLangOpts().OpenMP && FirstPart.isUsable()) {
      Actions.OpenMP().ActOnOpenMPLoopInitialization(ForLoc, FirstPart.get());
    }
  }

  // Set this only right before parsing the body to disallow break/continue in
  // the other parts.
  getCurScope()->EnterLoopBody(PrecedingLabel);

  // C99 6.8.5p5 - In C99, the body of the for statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.5p2:
  // The substatement in an iteration-statement implicitly defines a local scope
  // which is entered and exited each time through the loop.
  //
  // See comments in ParseIfStatement for why we create a scope for
  // for-init-statement/condition and a new scope for substatement in C++.
  //
  ParseScope InnerScope(this, Scope::DeclScope, C99orCXXorObjC,
                        Tok.is(tok::l_brace));

  // The body of the for loop has the same local mangling number as the
  // for-init-statement.
  // It will only be incremented if the body contains other things that would
  // normally increment the mangling number (like a compound statement).
  if (C99orCXXorObjC)
    getCurScope()->decrementMSManglingNumber();

  // Read the body statement.
  StmtResult Body(ParseStatement(TrailingElseLoc));

  // Pop the body scope if needed.
  InnerScope.Exit();

  // Leave the for-scope.
  ForallScope.Exit();

  if (Body.isInvalid())
    return StmtError();

  if (ForEach)
    return Actions.ObjC().FinishObjCForCollectionStmt(ForEachStmt.get(),
                                                      Body.get());

  if (ForRangeInfo.ParsedForRangeDecl())
    return Actions.FinishCXXForallRangeStmt(ForallRangeStmt.get(), Body.get());

  return Actions.ActOnForallStmt(ForLoc, T.getOpenLocation(), FirstPart.get(),
                                 SecondPart, ThirdPart, T.getCloseLocation(),
                                 Body.get());
}

void Parser::ParseKitsuneMemAccessQualifiers(ParsedAttributes &Attrs) {
  IdentifierInfo *AttrName = Tok.getIdentifierInfo();
  SourceLocation AttrNameLoc = Tok.getLocation();
  Attrs.addNew(AttrName, AttrNameLoc, AttributeScopeInfo(), nullptr, 0,
               Tok.getKind());
}
