add_llvm_component_library(LLVMTapirCommon
  TapirCommandLineUtils.cpp
  TapirStringUtils.cpp
  TapirTargetOptions.cpp

  # LLVM discourages the use of this, but it is used in some parts of LLVM for
  # good reason. Most of the Tapir code is in llvm/lib/Transforms, but a lot of
  # it should probably be moved elsewhere, including the code that is shared
  # with clang, flang and lld. For now, just do what we have to here.
  PARTIAL_SOURCES_INTENDED

  COMPONENT_NAME
  TapirCommon

  LINK_COMPONENTS
  Core
  Support
)

set(SOURCES
  DRFScopedNoAliasAA.cpp
  GPUABIOptions.cpp
  LoopSpawningTI.cpp
  LoopStripMine.cpp
  LoopStripMinePass.cpp
  LoweringUtils.cpp
  Outline.cpp
  GPUABI.cpp
  SerialABI.cpp
  SerializeSmallTasks.cpp
  TapirGPUUtils.cpp
  TapirLoopInfo.cpp
  TapirToTarget.cpp
)

if (KITSUNE_CUDA_ENABLED)
  list(APPEND SOURCES CudaABI.cpp)
endif ()

if (KITSUNE_HIP_ENABLED)
  list(APPEND SOURCES HipABI.cpp)
endif ()

if (KITSUNE_LAMBDA_ENABLED)
  list(APPEND SOURCES LambdaABI.cpp)
endif ()

if (KITSUNE_OMPTASK_ENABLED)
  list(APPEND SOURCES OMPTaskABI.cpp)
endif ()

if (KITSUNE_OPENCILK_ENABLED)
  list(APPEND SOURCES OpenCilkABI.cpp)
endif ()

if (KITSUNE_OPENMP_ENABLED)
  list(APPEND SOURCES OpenMPABI.cpp)
endif ()

if (KITSUNE_QTHREADS_ENABLED)
  list(APPEND SOURCES QthreadsABI.cpp)
endif ()

if (KITSUNE_REALM_ENABLED)
  list(APPEND SOURCES RealmABI.cpp)
endif ()

add_llvm_component_library(LLVMTapirOpts
  ${SOURCES}

  # LLVM discourages the use of this, but we definitely don't want to build
  # ABI targets unnecessarily (besides which, getting them to compile can be
  # fairly difficult without target-specific headers such as those from a Cuda
  # installation which may be in a non-standard location). Plus, they use them
  # in certain places anyway.
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Tapir

  DEPENDS
  intrinsics_gen
  LLVMLinker
  LLVMBitWriter

  COMPONENT_NAME
  TapirOpts

  LINK_COMPONENTS
  Analysis
  Core
  IPO
  IRReader
  Linker
  MC
  Scalar
  Support
  TapirCommon
  TransformUtils
  Vectorize
  Passes
)
