
include(AddClang)

# LLVM expects all files in directory to be used and combined into a single
# library. We could split the files here into subdirectories, but that seems
# like excessive organization when there are only a handful of files, and that
# number is unlikely to increase significantly (unless we decide to put in a lot
# more effort into the language). Instead, we create multiple libraries from the
# files here. Therefore, we need PARTIAL_SOURCES_INTENDED for each library that
# is created.

set(LLVM_LINK_COMPONENTS
  KitCore
  KitSupport
  Support)

# This library contains support code for Kitsnue's command line options and
# their interaction with the driver.
add_clang_library(clangKitDriver
  KitDriverUtils.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  kitsune_generated
  ClangDriverOptions

  # We must not link clangDriver here because that would create a circular
  # dependence between this and clangDriver.
  LINK_LIBS
  clangBasic)

# This contains some utilities for clang's AST, as well as some other support
# code. This is intended to be shared between the various parts of clang -
# specifically Sema and CodeGen. Most of the core code still lives in Sema/
# and CodeGen/.
add_clang_library(clangKitAST
  ASTUtils.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  kitsune_generated
  target_parser_gen

  LINK_LIBS
  clangAST
  clangBasic)
