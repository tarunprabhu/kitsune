# ${KITSUNE_SOURCE_DIR} is set to the kitsune/ directory at the top level of the
# llvm-project repo. ${KITSUNE_SOURCE_DIR}/lib/Clang contains utilities to
# handle Kitsune's extensions to C/C++, as well as support code for the driver
# and frontend.
#
# We must descend into the Clang subdirectory of ${KITSUNE_SOURCE_DIR}/lib
# explicitly because CMakeLists.txt in ${KITSUNE_SOURCE_DIR}/lib will not do so.
# That file will only descend into the subdirectories that are to be compiled
# as LLVM libraries to be linked to libLLVM.so.
add_subdirectory(${KITSUNE_SOURCE_DIR}/lib/Clang ${KITSUNE_BINARY_DIR}/lib/Clang)
add_subdirectory(Headers)
add_subdirectory(Basic)
add_subdirectory(APINotes)
add_subdirectory(Lex)
add_subdirectory(Parse)
add_subdirectory(AST)
add_subdirectory(ASTMatchers)
add_subdirectory(CrossTU)
add_subdirectory(Sema)
add_subdirectory(CodeGen)
add_subdirectory(Analysis)
add_subdirectory(Edit)
add_subdirectory(ExtractAPI)
add_subdirectory(Rewrite)
add_subdirectory(Driver)
add_subdirectory(Serialization)
add_subdirectory(Frontend)
add_subdirectory(FrontendTool)
add_subdirectory(Tooling)
add_subdirectory(DirectoryWatcher)
add_subdirectory(Index)
add_subdirectory(IndexSerialization)
add_subdirectory(InstallAPI)
add_subdirectory(StaticAnalyzer)
add_subdirectory(Format)
if(CLANG_INCLUDE_TESTS)
  add_subdirectory(Testing)
endif()
add_subdirectory(Interpreter)
add_subdirectory(Support)

if(CLANG_ENABLE_CIR)
  add_subdirectory(CIR)
endif()
