- conceptual
  - add CXXForallRangeStmt stuff to 
    - Parse/ParseStmt.cpp
  - add warnings, errors, and diagnostics
    - include/clang/Basic/DiagnosticParseKinds.td
    - include/clang/Basic/DiagnosticSemaKinds.td
  - add semantic stuff
    - include/clang/Sema/Sema.h
    - lib/Sema/SemaStmt.cpp
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

Files changed by type:

- random headers
  - include/clang-c/Index.h

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

- Parse
  - Parse/ParseStmt.cpp

- Sema
  - include/clang/Sema/Sema.h
  - Sema/SemaStmt.cpp (BuildCXXForRangeStmt has lots of code, is where most of the logic lies)
  - Sema/TreeTransform.h

- Serialization
  - Serialization/ASTBitCodes.h
  - Serialization/ASTReaderStmt.cpp
  - Serialization/ASTWriterStmt.cpp