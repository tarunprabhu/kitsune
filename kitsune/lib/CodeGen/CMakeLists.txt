add_llvm_component_library(LLVMKitCodeGen
  CGFBCuda.cpp
  CGFBHip.cpp
  CodeGenFatBinaries.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  KitAnalysis
  KitCore
  KitSupport
  Support
  )
