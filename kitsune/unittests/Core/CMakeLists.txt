set(LLVM_LINK_COMPONENTS
  AsmParser
  KitCore
  Core
)

add_kitsune_unittest(KitCoreTests
  ArgAttrsTest.cpp
  ArgUtilsTest.cpp
  AttrsImplTest.cpp
  BasicBlockUtilsTest.cpp
  ConstantUtilsTest.cpp
  DiagnosticsTest.cpp
  DIUtilsTest.cpp
  EmbUtilsTest.cpp
  FuncAttrsTest.cpp
  FuncUtilsTest.cpp
  GVAttrsTest.cpp
  GVUtilsTest.cpp
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
  TTUtilsTest.cpp
  ValueUtilsTest.cpp
)

target_include_directories(KitCoreTests PUBLIC
  ${KITSUNE_SOURCE_DIR}/lib)
