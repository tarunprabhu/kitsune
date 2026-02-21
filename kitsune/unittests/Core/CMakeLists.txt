set(LLVM_LINK_COMPONENTS
  AsmParser
  KitCore
  Core
)

add_kitsune_unittest_with_input_files(KitCoreTests
  ConstantUtilsTest.cpp
  EmbUtilsTest.cpp
  FunctionUtilsTest.cpp
  GlobalVariableUtilsTest.cpp
  IntrinsicUtilsTest.cpp
  LoopAttrsTest.cpp
  LoopUtilsTest.cpp
  KernelPropertiesTest.cpp
  ModuleUtilsTest.cpp
  TapirCoreTest.cpp
  TypeUtilsTest.cpp

  EXPORT_SYMBOLS
)

add_dependencies(KitCoreTests intrinsics_gen)

target_link_libraries(KitCoreTests PRIVATE LLVMTestingSupport)
