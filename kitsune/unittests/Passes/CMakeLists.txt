set(LLVM_LINK_COMPONENTS
  AsmParser
  Core
  KitCore
  KitPasses
  Passes
)

add_kitsune_unittest_with_input_files(KitPassesTests
  ExtensionPointsTest.cpp
  PipelineUtilsTest.cpp

  EXPORT_SYMBOLS
)

add_dependencies(KitPassesTests intrinsics_gen)

target_link_libraries(KitPassesTests PRIVATE LLVMTestingSupport)
