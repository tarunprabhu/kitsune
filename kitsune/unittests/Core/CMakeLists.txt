set(LLVM_LINK_COMPONENTS
  AsmParser
  KitCore
  Core
)

add_kitsune_unittest_with_input_files(KitCoreTests
  ConstantUtilsTest.cpp
  DIUtilsTest.cpp
  EmbUtilsTest.cpp
  FuncAttrsTest.cpp
  FunctionUtilsTest.cpp
  GVAttrsTest.cpp
  InstAttrsTest.cpp
  InstructionUtilsTest.cpp
  IntrinsicUtilsTest.cpp
  LoopAttrsTest.cpp
  LoopUtilsTest.cpp
  KernelPropertiesTest.cpp
  MetadataUtilsTest.cpp
  ModuleAttrsTest.cpp
  ModuleUtilsTest.cpp
  TapirCoreTest.cpp
  TypeUtilsTest.cpp
  ValueUtilsTest.cpp

  EXPORT_SYMBOLS
)

target_link_libraries(KitCoreTests PRIVATE LLVMTestingSupport)
