add_llvm_component_library(LLVMKitPasses
  PassUtils.cpp
  PipelineUtils.cpp

  LINK_COMPONENTS
  Core
  KitAnalysis
  KitCodeGen
  KitTransforms
)
