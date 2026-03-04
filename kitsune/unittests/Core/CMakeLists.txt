set(LLVM_LINK_COMPONENTS
  AsmParser
  KitCore
  Core
)

add_kitsune_unittest_with_input_files(KitCoreTests
  ConstantUtilsTest.cpp
  DIUtilsTest.cpp
  EmbUtilsTest.cpp
  FunctionUtilsTest.cpp
  GlobalVariableUtilsTest.cpp
  InstructionUtilsTest.cpp
  IntrinsicUtilsTest.cpp
  LoopAttrsTest.cpp
  LoopUtilsTest.cpp
  KernelPropertiesTest.cpp
  ModuleUtilsTest.cpp
  TapirCoreTest.cpp
  TypeUtilsTest.cpp

  EXPORT_SYMBOLS
)

target_link_libraries(KitCoreTests PRIVATE LLVMTestingSupport)
