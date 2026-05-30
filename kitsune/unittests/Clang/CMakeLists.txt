set(LLVM_LINK_COMPONENTS
  Core
)

add_kitsune_unittest_with_input_files(KitClangTests
  ReductionUtilsTest.cpp

  EXPORT_SYMBOLS
)

# FIXME: Instead of explicitly linking the libraries here, add support in
# add_kitsune_unittest to do so.
target_link_libraries(KitClangTests PRIVATE clangKitSupport)

target_link_libraries(KitClangTests PRIVATE LLVMTestingSupport)
