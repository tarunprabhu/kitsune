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

add_kitsune_unittest(KitPasses
  PipelineUtilsTest.cpp
)
