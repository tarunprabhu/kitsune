- conceptual
  - add CXXForallRangeStmt stuff to 
    - Parse/ParseStmt.cpp
  - add warnings, errors, and diagnostics
    - include/clang/Basic/DiagnosticParseKinds.td
    - include/clang/Basic/DiagnosticSemaKinds.td
  - add semantic stuff
    - include/clang/Sema/Sema.h
    - Sema/SemaStmt.cpp
    - Sema/AnalysisBasedWarnings.cpp
    - Sema/SemaDeclAttr.cpp
  - add the CXXForallRangeStmt
    - Basic/StmtNodes.td
    - include/clang/AST/StmtCXX.h
    - AST/StmtCXX.cpp
    - include/clang-c/Index.h
    - include/clang/AST/RecursiveASTVisitor.h
    - Sema/TreeTransform.h
  - ASTMatchers stuff
    - ASTMatchers/ASTMatchers.h
    - ASTMatchers/ASTMatchersInternal.cpp  
    - ASTMatchers/Dynamic/Registry.cpp
    - Serialization/ASTBitCodes.h
  - AST stuff
    - AST/ASTImporter.cpp
    - AST/ExprConstant.cpp
    - AST/StmtPrinter.cpp
    - AST/StmtProfile.cpp
  - Serialization stuff
    - Serialization/ASTReaderStmt.cpp
    - Serialization/ASTWriterStmt.cpp
  - compiles without errors or warnings
  - analysis stuff
    - Analysis/CFG.cpp
    - Analysis/ExprMutationAnalyzer.cpp
  - codegen stuff
    - CodeGen/CGStmt.cpp
    - CodeGen/CodeGenFunction.cpp
    - CodeGen/CodeGenFunction.h
    - CodeGen/CodeGenPGO.cpp
    - CodeGen/CoverageMappingGen.cpp
  - fix CXXForRangeStmt misses

Searches:
  - CXXForRangeStmt
  - cxxForRangeStmt (case sensitive)
  - hasLoopVariable (AST matcher)
  - hasRangeInit (AST matcher)
  - findRangeLoopMutation
  - RebuildForRangeWithDereference (function argument is CXXForRangeStmt)
  - DiagnoseForallRangeVariableCopies (function argument is CXXForRangeStmt)


Files changed by type:

- random headers
  - include/clang-c/Index.h

- Analysis
  - Analysis/CFG.cpp
  - Analysis/ExprMutationAnalyzer.cpp

- AST
  - /include/clang/AST/StmtCXX.h
  - AST/StmtCXX.cpp
  - include/clang/AST/RecursiveASTVisitor.h
  - AST/ASTImporter.cpp
  - AST/ExprConstant.cpp
  - AST/StmtPrinter.cpp
  - AST/StmtProfile.cpp

- ASTMatchers
  - ASTMatchers/ASTMatchers.h
  - ASTMatchers/ASTMatchersInternal.cpp
  - ASTMatchers/Dynamic/Registry.cpp

- Basic
  - Basic/DiagnosticParseKinds.td
  - Basic/DiagnosticSemaKinds.td
  - Basic/StmtNodes.td

- CodeGen
  - CodeGen/CGStmt.cpp (has EmitCXXForRangeStmt)
  - CodeGen/CodeGenFunction.cpp
  - CodeGen/CodeGenFunction.h
  - CodeGen/CodeGenPGO.cpp
  - CodeGen/CoverageMappingGen.cpp

- Parse
  - Parse/ParseStmt.cpp

- Sema
  - include/clang/Sema/Sema.h
  - Sema/SemaStmt.cpp (BuildCXXForRangeStmt has lots of code, is where most of the logic lies)
  - Sema/TreeTransform.h
  - Sema/AnalysisBasedWarnings.cpp
  - Sema/SemaDeclAttr.cpp
  - Sema/SemaDeclCXX.cpp
  - Sema/SemaStmtAttr.cpp

- Serialization
  - include/clang/Serialization/ASTBitCodes.h
  - Serialization/ASTReaderStmt.cpp
  - Serialization/ASTWriterStmt.cpp

- StaticAnalyzer
  - StaticAnalyzer/Core/BugReporter.cpp
  - StaticAnalyzer/Core/CoreEngine.cpp
  - StaticAnalyzer/Core/ExprEngine.cpp

- Tooling
  - Tooling/Refactoring/Extract/SourceExtraction.cpp

- tools
  - tools/libclang/CIndex.cpp
  - tools/libclang/CXCursor.cpp