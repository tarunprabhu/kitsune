add_llvm_component_library(LLVMKitTransforms
  DeLICM.cpp
  EarlyAnnotate.cpp
  EmbLinkLibDeviceBitcode.cpp
  EmbLowerKitIntrinsicsEarly.cpp
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
  GenerateCtorOpenCilk.cpp
  GenerateCtorOpenMP.cpp
  GenerateCtorPthreads.cpp
  GenerateCtorQthreads.cpp
  LoopWrapping.cpp
  LowerKitReduceIntrinsics.cpp
  NormalizeLoopControlBlocks.cpp
  PreLowerAnnotate.cpp
  PrefetchForDevice.cpp
  PrepareParallelLoops.cpp
  PrepareReductionLoops.cpp
  PrepareTapirLoops.cpp
  RecomputeKernelProperties.cpp
  SecondaryIVElimination.cpp
  Serialize.cpp

  DEPENDS
  intrinsics_gen
  kitsune_generated

  # The pass that optimizes embedded modules constructs a private pass pipeline
  # and runs a standard sequence of optimization passes on the module. This
  # requires a PassBuilder which is provided by libLLVMPasses. However,
  # libLLVMPasses requires this library, libLLVMKitTransforms, to instantiate
  # the passes when constructing the pass pipeline, thus resulting in a circular
  # dependence.
  #
  #     libLLVMPasses => libLLVMKitTransforms => libLLVMPasses
  #
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
)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  target_link_options(LLVMKitTransforms PUBLIC
    LINKER:-U,_clRefineLaunches,-U,_clUseYLaunch)
endif ()
