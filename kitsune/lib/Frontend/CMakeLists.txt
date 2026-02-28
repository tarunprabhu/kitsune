add_llvm_component_library(LLVMKitFrontend
  CommandLineOptions.cpp
  Diagnostics.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  KitSupport
  Support
)
