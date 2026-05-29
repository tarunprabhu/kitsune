set(LLVM_LINK_COMPONENTS
  Support
)

add_kitsune_unittest_with_input_files(KitConfigTests
  ConfigTest.cpp

  EXPORT_SYMBOLS
)

target_link_libraries(KitConfigTests PRIVATE LLVMTestingSupport)
