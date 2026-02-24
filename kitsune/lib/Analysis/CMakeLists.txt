add_llvm_component_library(LLVMKitAnalysis
  TapirLoopNestAnalysis.cpp
  TapirTargetAnalysis.cpp

  DEPENDS
  intrinsics_gen

  # Including TapirOpts here results in a circular dependence since TapirOpts
  # depends on KitAnalysis. The issue is that the TapirTargetAnalysis pass
  # creates TapirTarget instances, several of which are defined in TapirOpts.
  # This feels like one of those dependences that could be broken, but it is not
  # immediately apparent how.
  LINK_COMPONENTS
  Analysis
  Core
  KitCore
  KitSupport
  KitTargets
  Support
  TapirOpts
)
