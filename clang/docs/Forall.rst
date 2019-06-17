- Add "forall" keyword to StmtNodes.td
- Add ForallStmt to Stmt.h (lots of stuff here)
- Add methods to AST/Stmt.cpp
  - constructor
  - getConditionVariable
  - setConditionVariable
- Add ForallStmt bitfields
- Add "forall" token to TokenKinds.def
- Add DEF_TRAVERSE_STMT(ForallStmt, {}) to RecursiveASTVisitor.h
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
- Add visitor to AST/StmtPrinter.cpp
- Add visitor to AST/StmtProfile.cpp
- Add visitor to Serialization/ASTReaderStmt.cpp
- Add visitor to Serialization/ASTWriterStmt.cpp
- Add STMT_FORALL to ASTBitCodes.h
- Add CXCursor support
  - clang-c/Index.h
  - libclang/CIndex.cpp
  - libclang/CXCursor.cpp

  At this point the code should compile, but not necessary pass tests.

- Add AST matchers (search case sensitive on "forStmt", trace forStmt, hasIncrement, hasLoopInit)
  - ASTMatchers/ASTMatchers.h
  - Core/CheckerManager.h
  - ASTMatchers/Dynamic/Registry.cpp
  - StaticAnalyzer/Core/LoopUnrolling.cpp (need to trace forLoopMatcher
  - ASTMatchers/ASTMatchersInternal.cpp
  - StaticAnalyzer/ExprEngine.cpp
  - AST/ASTImporter.cpp
  - AST/ExprConstant.cpp
  - AST/ParentMap.cpp

  Code compiles again

- Add Analysis stuff
  - Analysis/CFG.cpp
  - Analysis/LiveVariables.cpp
  - Analysis/ReachableCode.cpp

- CodeGen stuff
  - CodeGen/CGStmt.cpp (EmitForallStmt)
  - CodeGen/CodeGenFunction.cpp
  - CodeGen/CodeGenPGO.cpp
  - CodeGen/CoverageMappingGen.cpp

- Sema stuff
  - Sema/AnalysisBasesWarnings.cpp
  - Sema/SemaChecking.cpp
  - Sema/SemaDeclAttr.cpp
  - Sema/SemaDeclCXX.cpp
  - Sema/SemaStmt.cpp
  - Sema/SemaStmtAttr.cpp

- Random stuff
  - ARCMigrate/Transforms.cpp 
  - Frontend/RewriteModernObjC.cpp