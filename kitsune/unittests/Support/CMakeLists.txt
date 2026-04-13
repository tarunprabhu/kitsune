set(LLVM_LINK_COMPONENTS
  KitCore
  KitSupport
  Support
)

add_kitsune_unittest_with_input_files(KitSupportTests
  DiagnosticsTest.cpp
  FromIntTest.cpp
  FromStringTest.cpp
  TTIDUtilsTest.cpp

  EXPORT_SYMBOLS
)

target_link_libraries(KitSupportTests PRIVATE LLVMTestingSupport)
