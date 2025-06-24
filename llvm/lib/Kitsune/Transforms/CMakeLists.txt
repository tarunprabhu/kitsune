add_llvm_component_library(LLVMKitOpts
  EmbLinkLibDeviceBitcode.cpp
  EmbPrepare.cpp
  EmbOptimize.cpp
  EmbResolveLibDeviceCalls.cpp
  GenerateCtors.cpp
  GenerateCtorCuda.cpp
  GenerateCtorHip.cpp
  LowerKitsuneRuntimeIntrinsics.cpp
  LowerMobileIntrinsics.cpp
  RecomputeKernelProperties.cpp
  StripKitsuneAddrSpace.cpp

  DEPENDS
  intrinsics_gen

  # Including Passes here results in a circular dependence since Passes requires
  # KitOpts. This is a known issue and like a very fundamental one that would
  # be very difficult to fix. The pass that optimizes embedded modules
  # constructs a private pass pipeline in order to run a standard sequence of
  # optimization passses on the module. This requires a PassBuilder which is
  # provided by Passes. However, Passes in turn requires this library, KitOpts,
  # in order to instantiate the passes when constructing the pipeline, thus
  # resulting in a circular dependence. Breaking this dependency cycle would
  # probably entail reorganizing the pass-related libraries which is a very
  # deep change. The fundamental issue is that LLVM was never designed with the
  # ability for passes to construct their own pipelines. For now, we accept that
  # this circular dependence exists and disallow building with shared libraries.
  LINK_COMPONENTS
  Analysis
  Core
  IRReader
  KitAnalysis
  KitCore
  KitSupport
  Linker
  Passes
  Support
  KitTransformUtils
)
