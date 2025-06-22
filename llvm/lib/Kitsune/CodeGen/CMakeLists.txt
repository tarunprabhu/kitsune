add_llvm_component_library(LLVMKitCodeGen
  CodeGenFatBinaries.cpp
  CodeGenFatBinariesCuda.cpp
  CodeGenFatBinariesHip.cpp

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
