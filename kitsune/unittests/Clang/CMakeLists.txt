set(LLVM_LINK_COMPONENTS
  Core
)

set(CLANG_LINK_COMPONENTS
  KitAST
)

add_kitsune_unittest(KitClangTests
  ReductionUtilsTest.cpp
)
