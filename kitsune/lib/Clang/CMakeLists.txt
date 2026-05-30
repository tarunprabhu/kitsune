include(AddClang)

set(LLVM_LINK_COMPONENTS
  KitCore
  KitSupport
  Support)

add_clang_library(clangKitSupport
  ReductionUtils.cpp

  DEPENDS
  kitsune_generated
  target_parser_gen
  ClangDriverOptions

  LINK_LIBS
  clangAST
  clangBasic)
