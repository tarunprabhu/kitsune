add_llvm_component_library(LLVMKitFrontend
  CommandLineOptions.cpp

  DEPENDS
  intrinsics_gen
  kitsune_generated

  LINK_COMPONENTS
  Analysis
  Core
  KitSupport
  Support
)
