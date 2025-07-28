set(LLVM_LINK_COMPONENTS
  AsmParser
  KitCore
  Core
)

add_kitsune_unittest_with_input_files(KitCoreTests
  ConstantUtilsTest.cpp
  EmbUtilsTest.cpp
  GlobalVariableUtilsTest.cpp
  IntrinsicUtilsTest.cpp
  KernelPropertiesTest.cpp
  ModuleUtilsTest.cpp
  SingletonUtilsTest.cpp
  TypeUtilsTest.cpp

  EXPORT_SYMBOLS
)

add_dependencies(KitCoreTests intrinsics_gen)

target_link_libraries(KitCoreTests PRIVATE LLVMTestingSupport)
