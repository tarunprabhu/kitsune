add_llvm_component_library(LLVMKitTransforms
  EarlyAnnotate.cpp
  EmbLinkLibDeviceBitcode.cpp
  EmbLowerKitIntrinsicsLibDevice.cpp
  EmbPrepare.cpp
  EmbPrepareCuda.cpp
  EmbPrepareHip.cpp
  EmbOptimize.cpp
  EmbResolveCallsCuda.cpp
  EmbResolveCallsHip.cpp
  EmbResolveLibDeviceCalls.cpp
  GenerateCtors.cpp
  GenerateCtorCuda.cpp
  GenerateCtorHip.cpp
  GenerateCtorOpenMP.cpp
  GenerateCtorPthreads.cpp
  GenerateCtorQthreads.cpp
  PreLowerAnnotate.cpp
  PrefetchForDevice.cpp
  RecomputeKernelProperties.cpp
  Serialize.cpp

  DEPENDS
  intrinsics_gen

  # The pass that optimizes embedded modules constructs a private pass pipeline
  # and runs a standard sequence of optimization passes on the module. This
  # requires a PassBuilder which is provided by Passes. However, Passes in turn
  # requires this library, KitTransforms, in order to instantiate the passes
  # when constructing the pipeline, thus resulting in a circular dependence.
  #
  #     libLLVMPasses => libLLVMKitTransforms => libLLVMPasses
  #
  LINK_COMPONENTS
  Analysis
  Core
  IRReader
  KitAnalysis
  KitCore
  KitFrontend
  KitSupport
  Linker
  Passes
  Support
)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  target_link_options(LLVMKitTransforms PUBLIC
    LINKER:-U,_clRefineLaunches,-U,_clUseYLaunch)
endif ()
