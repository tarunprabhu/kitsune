set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  AsmParser
  Core
  KitCore
  KitPasses
  Passes
)

add_kitsune_unittest_with_input_files(KitPassesTests
  PipelineUtilsTest.cpp

  EXPORT_SYMBOLS
)

target_link_libraries(KitPassesTests PRIVATE LLVMTestingSupport)
