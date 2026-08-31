add_llvm_component_library(LLVMKitCodeGen
  CGFBCuda.cpp
  CGFBHip.cpp
  CodeGenFatBinaries.cpp
  EmbLowerKitIntrinsics.cpp
  EmbModuleLegacyPass.cpp
  LowerIntrinsics.cpp
  StripKitAddrSpaces.cpp

  DEPENDS
  intrinsics_gen
  kitsune_generated

  LINK_COMPONENTS
  Analysis
  Core
  KitAnalysis
  KitCore
  KitSupport
  Support
)
