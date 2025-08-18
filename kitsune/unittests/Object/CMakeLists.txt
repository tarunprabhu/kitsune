set(LLVM_LINK_COMPONENTS
  KitObject
  KitSupport
  Object
  Support
)

add_kitsune_unittest_with_input_files(KitObjectTests
  ArchiveUtilsTest.cpp
  BinaryUtilsTest.cpp
  CheckUtils.cpp
  CompressedBinary.cpp
  EmbDeviceCodeTest.cpp
  EmbDeviceCodeContextTest.cpp
  ObjectUtilsTest.cpp

  EXPORT_SYMBOLS
)

target_link_libraries(KitObjectTests PRIVATE LLVMTestingSupport)
