Adding "forall" keyword logical steps

- Add "forall" keyword
  - Basic/StmtNodes.td
- Add ForallStmt 
  - AST/Stmt.h (lots of stuff here)
- Add methods to AST/Stmt.cpp
  - constructor
  - getConditionVariable
  - setConditionVariable
- Add ForallStmt bitfields
- Add "forall" token to TokenKinds.def
- Add DEF_TRAVERSE_STMT(ForallStmt, {}) to 
  - AST/RecursiveASTVisitor.h
- Add ParseForallStatement
  - Parser.h
  - Parse/ParseStmt.cpp
  - add errors to DiagnosticParseKinds.td
- Add TransformForallStmt to TreeTransform.h
  - transforms each part of the statement
  - rebuild, used for template expansion
- Add ActOnForallStmt
  - Sema.h
  - add semantic errors to DiagnosticSemaKinds.td
- Add visitor
  - AST/StmtPrinter.cpp
- Add visitors
  - AST/StmtProfile.cpp
  - Serialization/ASTReaderStmt.cpp
  - Serialization/ASTWriterStmt.cpp
- Add STMT_FORALL
  - ASTBitCodes.h
- Add CXCursor support
  - clang-c/Index.h
  - libclang/CIndex.cpp
  - libclang/CXCursor.cpp


At this point the code should compile, but not necessary pass tests.

Systematic search/duplicate code. 
- Search on:
  - ForStmt (72 files match)
  - forStmt (case sensitive to find AST matching stuff
  - _for (for errors and warnings)
  - hasIncrement
  - hasLoopInit
  - forLoopMatcher



Files that needed to be changed:

- Basic
  - Basic/StmtNodes.td

- Core
  - Core/CheckerManager.h

- Add AST matchers 
  - AST/RecursiveASTVisitor.h
  - AST/StmtNodes.td
  - ASTMatchers/ASTMatchers.h
  - ASTMatchers/Dynamic/Registry.cpp
  - ASTMatchers/ASTMatchersInternal.cpp
  - AST/ASTImporter.cpp
  - AST/ExprConstant.cpp
  - AST/ParentMap.cpp
  - AST/Stmt.cpp
  - AST/StmtPrinter.cpp
  - AST/StmtProfile.cpp


- Add Analysis stuff
  - Analysis/CFG.cpp
  - Analysis/LiveVariables.cpp
  - Analysis/ReachableCode.cpp

- CodeGen stuff
  - CodeGen/CGStmt.cpp (EmitForallStmt)
  - CodeGen/CodeGenFunction.cpp
  - CodeGen/CodeGenPGO.cpp
  - CodeGen/CoverageMappingGen.cpp

- libclang stuff
  - libclang/CIndex.cpp
  - libclang/CXCursor.cpp


- Parse stuff
  - Parse/ParseStmt.cpp

- Sema stuff
  - Sema/Sema.h
  - Sema/AnalysisBasesWarnings.cpp
  - Sema/SemaChecking.cpp
  - Sema/SemaDeclAttr.cpp
  - Sema/SemaDeclCXX.cpp
  - Sema/SemaStmt.cpp
  - Sema/SemaStmtAttr.cpp
  - Sema/TreeTransform.h

- Serialization
  - Serialization/ASTReaderStmt.cpp
  - Serialization/ASTWriterStmt.cpp
  - Serialization/ASTBitCodes.h

- StaticAnalyzer stuff
  - StaticAnalyzer/Checkers/CheckSecuritySyntaxOnly.cpp
  - StaticAnalyzer/Checkers/IdenticalExprChecker.cpp
  - StaticAnalyzer/Checkers/MallocOverflowSecurityChecker.cpp
  - StaticAnalyzer/Core/BugReporter.cpp
  - StaticAnalyzer/Core/CoreEngine.cpp
  - StaticAnalyzer/Core/ExprEngine.cpp
  - StaticAnalyzer/Core/LoopUnrolling.cpp (need to trace forLoopMatcher)
  - StaticAnalyzer/Core/LoopWidening.cpp
  - StaticAnalyzer/Core/PathDiagnostic.cpp

- Random stuff
  - ARCMigrate/Transforms.cpp 
  - Frontend/Rewrite/RewriteModernObjC.cpp
  - Frontend/Rewrite/RewriteObjC.cpp
  - Tooling/Refactoring/Extract/SourceExtraction.cpp

 