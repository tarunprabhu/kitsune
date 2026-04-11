set(LLVM_LINK_COMPONENTS
  AsmParser
  KitCore
  Core
)

add_kitsune_unittest_with_input_files(KitCoreTests
  ArgAttrsTest.cpp
  ArgUtilsTest.cpp
  AttrsImplTest.cpp
  ConstantUtilsTest.cpp
  DIUtilsTest.cpp
  EmbUtilsTest.cpp
  FuncAttrsTest.cpp
  FuncUtilsTest.cpp
  GVAttrsTest.cpp
  InstAttrsTest.cpp
  InstUtilsTest.cpp
  IntrinsicUtilsTest.cpp
  LoopAttrsTest.cpp
  LoopUtilsTest.cpp
  KernelPropertiesTest.cpp
  MetadataUtilsTest.cpp
  ModuleAttrsTest.cpp
  ModuleUtilsTest.cpp
  TapirCoreTest.cpp
  TestUtils.cpp
  TypeUtilsTest.cpp
  ValueUtilsTest.cpp

  EXPORT_SYMBOLS
)

target_include_directories(KitCoreTests PUBLIC
  ${KITSUNE_SOURCE_DIR}/lib)

target_link_libraries(KitCoreTests PRIVATE LLVMTestingSupport)
