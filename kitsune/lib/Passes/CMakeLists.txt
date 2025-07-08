add_llvm_component_library(LLVMKitPasses
  PipelineUtils.cpp

  LINK_COMPONENTS
  Core
  KitAnalysis
  KitTransforms
)
