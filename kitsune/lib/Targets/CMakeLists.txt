set(SOURCES
  GPUTTCommon.cpp
  GPUTTUtils.cpp
  SerialTT.cpp
  TapirTargets.cpp
)

if (KITSUNE_CUDA_ENABLED)
  list(APPEND SOURCES CudaABI.cpp)
endif ()

if (KITSUNE_HIP_ENABLED)
  list(APPEND SOURCES HipABI.cpp)
endif ()

if (KITSUNE_OPENMP_ENABLED)
  list(APPEND SOURCES OpenMPABI.cpp)
endif ()

if (KITSUNE_PTHREADS_ENABLED)
  list(APPEND SOURCES PthreadsTT.cpp)
endif ()

if (KITSUNE_QTHREADS_ENABLED)
  list(APPEND SOURCES QthreadsTT.cpp)
endif ()

if (KITSUNE_REALM_ENABLED)
  list(APPEND SOURCES RealmABI.cpp)
endif ()

add_llvm_component_library(LLVMKitTargets
  ${SOURCES}

  # LLVM discourages the use of this, but we definitely don't want to build
  # ABI targets unnecessarily (besides which, getting them to compile can be
  # fairly difficult without target-specific headers such as those from a Cuda
  # installation which may be in a non-standard location). Plus, they use them
  # in certain places anyway.
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  intrinsics_gen
  kitsune_generated

  # There is a circular dependence here between this library and LLVMCore. For
  # more details, see llvm/lib/IR/CMakeLists.txt. It would be nice if we could
  # get rid of these, but that is highly unlikely.
  LINK_COMPONENTS
  BitReader
  BitWriter
  Core
  KitFrontend
  KitSupport
  Support
  TapirOpts
  TargetParser
  TransformUtils
)
