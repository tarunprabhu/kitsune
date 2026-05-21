add_llvm_component_library(LLVMKitFrontend
  CommandLineOptions.cpp
  ReductionUtils.cpp

  DEPENDS
  intrinsics_gen
  kitsune_generated

  LINK_COMPONENTS
  Analysis
  Core
  KitCore
  KitSupport
  Support
)
