set(SOURCES
  DRFScopedNoAliasAA.cpp
  LoopSpawningTI.cpp
  LoopStripMine.cpp
  LoopStripMinePass.cpp
  LoweringUtils.cpp
  Outline.cpp
  SerialABI.cpp
  SerializeSmallTasks.cpp
  TapirLoopInfo.cpp
  TapirToTarget.cpp
)

if (KITSUNE_LAMBDA_ENABLED)
  list(APPEND SOURCES LambdaABI.cpp)
endif ()

if (KITSUNE_OMPTASK_ENABLED)
  list(APPEND SOURCES OMPTaskABI.cpp)
endif ()

if (KITSUNE_OPENCILK_ENABLED)
  list(APPEND SOURCES OpenCilkABI.cpp)
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
  kitsune_generated
  LLVMLinker
  LLVMBitWriter

  COMPONENT_NAME
  TapirOpts

  LINK_COMPONENTS
  Analysis
  Core
  Demangle
  IPO
  IRReader
  KitAnalysis
  KitCore
  Linker
  Scalar
  Support
  TargetParser
  TransformUtils
)
