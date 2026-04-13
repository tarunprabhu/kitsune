set(SOURCES
  GPUTTCommon.cpp
  GPUTTLoop.cpp
  OpenMPTT.cpp
  PthreadsTT.cpp
  SerialTT.cpp
  TapirTargets.cpp
)

if (KITSUNE_CUDA_ENABLED)
  list(APPEND SOURCES CudaABI.cpp)
endif ()

if (KITSUNE_HIP_ENABLED)
  list(APPEND SOURCES HipABI.cpp)
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
  # tapir targets unnecessarily. We could move each tapir target into its own
  # directory, then conditionally add that directory. That is, arguably, a
  # worse approach than this one. Unconditionally building tapir targets can
  # also be challenging since some may require specific targets to have been
  # enabled, or may utilize external header files that may not be in the
  # standard locations.
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
