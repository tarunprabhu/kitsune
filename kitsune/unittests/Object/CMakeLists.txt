set(LLVM_LINK_COMPONENTS
  KitObject
  KitSupport
  Object
  Support
)

add_kitsune_unittest_with_input_files(KitObjectTests
  BinaryUtilsTest.cpp
  # EmbDeviceCodeTest.cpp
  # ObjectUtilsTest.cpp

  # FIXME: Remove this when we sort everything out
  PARTIAL_SOURCES_INTENDED

  EXPORT_SYMBOLS
)

target_link_libraries(KitObjectTests PRIVATE LLVMTestingSupport)
