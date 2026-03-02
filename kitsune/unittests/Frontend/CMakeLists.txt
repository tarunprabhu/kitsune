set(LLVM_LINK_COMPONENTS
  KitFrontend
)

add_kitsune_unittest_with_input_files(KitFrontendTests
  DiagnosticsTest.cpp

  EXPORT_SYMBOLS
)

target_link_libraries(KitCoreTests PRIVATE LLVMTestingSupport)
