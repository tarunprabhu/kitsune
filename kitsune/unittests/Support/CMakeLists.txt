set(LLVM_LINK_COMPONENTS
  KitSupport
  Support
)

add_kitsune_unittest_with_input_files(KitSupportTests
  TTIDUtilsTest.cpp

  EXPORT_SYMBOLS
)

target_link_libraries(KitSupportTests PRIVATE LLVMTestingSupport)
