add_llvm_component_library(LLVMKitAnalysis
  EarlyVerification.cpp
  PreLowerVerification.cpp
  TapirLoopNestAnalysis.cpp
  TTObjectsAnalysis.cpp

  DEPENDS
  intrinsics_gen
  kitsune_generated

  # Including TapirOpts here results in a circular dependence since TapirOpts
  # depends on KitAnalysis. The issue is that the TapirTargetAnalysis pass
  # creates TapirTarget instances, several of which are defined in TapirOpts.
  # The basic issue here is that most tapir utilities are in libLLVMTransforms,
  # but we need them for analysis. In many cases, it is absolutely clear that
  # those utilities should not be in libLLVMTransforms - but that would require
  # a significant amount of code reorganization that would cause greater
  # divergence from upstream.
  LINK_COMPONENTS
  Analysis
  Core
  KitCore
  KitSupport
  KitTargets
  Support
  TapirOpts
  TransformUtils
)
