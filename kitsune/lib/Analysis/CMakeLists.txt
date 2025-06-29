add_llvm_component_library(LLVMKitAnalysis
  TapirTargetAnalysis.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  KitCore
  KitSupport
  Support
)
