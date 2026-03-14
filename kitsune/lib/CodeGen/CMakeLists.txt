add_llvm_component_library(LLVMKitCodeGen
  CGFBCuda.cpp
  CGFBHip.cpp
  CodeGenFatBinaries.cpp
  EmbLowerKitIntrinsics.cpp
  EmbModuleLegacyPass.cpp
  LowerKitIntrinsics.cpp
  StripKitAddrSpaces.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  KitAnalysis
  KitCore
  KitFrontend
  KitSupport
  Support
)
