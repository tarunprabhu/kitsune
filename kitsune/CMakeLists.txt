#
# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune+tapir project.  It is released under
# the LLVM license.
#
cmake_minimum_required(VERSION 3.20)
project(kitsune C CXX)

include(ExternalProject)
include(FetchContent)

if (WIN32)
  message(FATAL_ERROR "Kitsune is not supported on Windows")
endif ()

set(KITSUNE_C_FRONTEND kitcc)
set(KITSUNE_CXX_FRONTEND kit++)
set(KITSUNE_Fortran_FRONTEND kitfort)

option(KITSUNE_INCLUDE_TESTS
  "Generate build targets for the Kitsune tests"
  ${LLVM_INCLUDE_TESTS})

option(KITSUNE_KOKKOS_ENABLE
  "Enable custom recognition and compilation of Kokkos"
  ON)

set(KITSUNE_CUDA_ENABLE OFF CACHE INTERNAL "Enable the cuda tapir target" FORCE)
set(KITSUNE_HIP_ENABLE OFF CACHE INTERNAL "Enable the hip tapir target" FORCE)
set(KITSUNE_OPENCILK_ENABLE OFF CACHE INTERNAL "Enable the opencilk tapir target" FORCE)
set(KITSUNE_OPENMP_ENABLE OFF CACHE INTERNAL "Enable the openmp tapir target" FORCE)
set(KITSUNE_QTHREADS_ENABLE OFF CACHE INTERNAL "Enable the qthreads tapir target" FORCE)
set(KITSUNE_REALM_ENABLE OFF CACHE INTERNAL "Enable the realm tapir target" FORCE)

# TODO: The _MAX variables are not currently used but it would be good to
# require a range instead of just a minimum. In practice, we may want to
# restrict it to just a single version because maintaining even a small range
# is likely to be a pain.
set(KITSUNE_CUDA_VERSION_MIN 12.3 CACHE INTERNAL "Minimum Cuda version required")
set(KITSUNE_CUDA_VERSION_MAX 12.3 CACHE INTERNAL "Maxmimum Cuda version supported")
set(KITSUNE_ROCM_VERSION_MIN 7.0 CACHE INTERNAL "Minimum ROCM version required")
set(KITSUNE_ROCM_VERSION_MAX 7.0 CACHE INTERNAL "Maximum ROCM version supported")

set(KITSUNE_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(KITSUNE_INCLUDE_DIR ${KITSUNE_SOURCE_DIR}/include)
set(KITSUNE_BINARY_DIR  ${PROJECT_BINARY_DIR})
set(KITSUNE_TARGETS_BINARY_DIR ${KITSUNE_BINARY_DIR}/targets)

# TODO: Need to update the openmp, qthreads and realm targets and add them
# back into this list. These targets have not been tested in a while and have
# probably bit-rotted. Enabling them will raise a configure-time error.
set(KITSUNE_ALL_TAPIR_TARGETS
  "cuda;hip;opencilk"
  CACHE STRING
  "All known Tapir targets.")

set(KITSUNE_DEFAULT_TAPIR_TARGETS
  "cuda;hip;opencilk"
  CACHE STRING
  "The Tapir targets that are built by default if targets are not explicitly specified.")

set(KITSUNE_ENABLE_TAPIR_TARGETS
  "${KITSUNE_DEFAULT_TAPIR_TARGETS}"
  CACHE STRING
  "Semicolon-separated list of runtime targets to build (or \"all\".)")

if (NOT KITSUNE_ENABLE_TAPIR_TARGETS)
  message(FATAL_ERROR "At least one runtime target must be built")
endif ()

foreach(target IN LISTS KITSUNE_ENABLE_TAPIR_TARGETS)
  if (target IN_LIST KITSUNE_ALL_TAPIR_TARGETS)
    message(STATUS "${target} tapir target enabled")
    string(TOUPPER "${target}" upper_target)
    set(KITSUNE_${upper_target}_ENABLE ON CACHE INTERNAL "Enable the ${target} tapir target" FORCE)
  else ()
    message(FATAL_ERROR "Unknown Tapir target: ${target}")
  endif()
endforeach()

if (KITSUNE_KOKKOS_ENABLE)
  # We currently don't support using a pre-built Kokkos - even if it has been
  # patched the way we need it to be. This could possibly be moved to
  # kitsune/runtime, just so everything that needs to be fetched and built is
  # there, even if this is not technically a runtime.

  set(KITSUNE_KOKKOS_SRC_DIR  ${KITSUNE_TARGETS_BINARY_DIR}/kokkos/kokkos)
  set(KITSUNE_KOKKOS_BUILD_DIR ${KITSUNE_TARGETS_BINARY_DIR}/kokkos/build)
  set(KITSUNE_KOKKOS_BUILD_ARGS
    -DKokkos_ENABLE_SERIAL=ON -DBUILD_TESTING=OFF
    -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release)

  # We have to patch kokkos to disable some sections of code that are using
  # exceptions.
  set(KITSUNE_KOKKOS_PATCH_EXCS ${KITSUNE_SOURCE_DIR}/patches/Kokkos_Exceptions.patch)

  #
  # TODO: Look for newer Kokkos updates.
  #
  message(STATUS "Checkout Kokkos into ${KITSUNE_KOKKOS_SRC_DIR}")
  FetchContent_Declare(
    kokkos
    GIT_REPOSITORY      https://github.com/kokkos/kokkos.git
    GIT_TAG             606866d # 4.0.00
    SOURCE_DIR          ${KITSUNE_KOKKOS_SRC_DIR}
    BINARY_DIR          ${KITSUNE_KOKKOS_BUILD_DIR}
    CMAKE_ARGS          ${KITSUNE_KOKKOS_BUILD_ARGS}
    PATCH_COMMAND       patch -p0 --verbose --input=${KITSUNE_KOKKOS_PATCH_EXCS}
    UPDATE_DISCONNECTED FALSE)

  FetchContent_MakeAvailable(kokkos)

  set(KITSUNE_KOKKOS_DEFAULT_PREPROCESSOR_FLAGS
    "-I${CMAKE_INSTALL_PREFIX}/include -I${CMAKE_INSTALL_PREFIX}/include/kokkos")

  set(KITSUNE_KOKKOS_DEFAULT_COMPILER_FLAGS
    "")

  set(KITSUNE_KOKKOS_DEFAULT_LINKER_FLAGS
    "-L${CMAKE_INSTALL_PREFIX}/lib -lkokkoscore -ldl")

  set(KITSUNE_KOKKOS_EXTRA_PREPROCESSOR_FLAGS
    ""
    CACHE STRING
    "Additional preprocessor flags needed for Kokkos support")

  set(KITSUNE_KOKKOS_EXTRA_COMPILER_FLAGS
    ""
    CACHE STRING
    "Additional compiler flags needed for Kokkos support")

  set(KITSUNE_KOKKOS_EXTRA_LINKER_FLAGS
    ""
    CACHE STRING
    "Additional linker flags needed for Kokkos support")
endif()

if (KITSUNE_CUDA_ENABLE)
  find_package(CUDAToolkit ${KITSUNE_CUDA_VERSION_MIN} REQUIRED)

  find_library(LIB_CUDART cudart REQUIRED
    PATHS ${CUDAToolkit_LIBRARY_DIR}
    NO_DEFAULT_PATH)

  find_library(LIB_CUDA cuda REQUIRED
    PATHS ${CUDAToolkit_LIBRARY_DIR}/stubs
    NO_DEFAULT_PATH)

  set(KITSUNE_CUDA_DEFAULT_PREPROCESSOR_FLAGS
    "-D_tapir_cuda_target")

  set(KITSUNE_CUDA_DEFAULT_COMPILER_FLAGS
    "")

  set(KITSUNE_CUDA_DEFAULT_LINKER_FLAGS
    "-L${CUDAToolkit_LIBRARY_DIR} -L${CUDAToolkit_LIBRARY_DIR}/stubs -lcudart -lcuda -lkitrt")

  set(KITSUNE_CUDA_EXTRA_PREPROCESSOR_FLAGS
    ""
    CACHE STRING
    "Additional preprocessor flags needed for the Cuda target")

  set(KITSUNE_CUDA_EXTRA_COMPILER_FLAGS
    ""
    CACHE STRING
    "Additional compiler flags needed for the Cuda target")

  set(KITSUNE_CUDA_EXTRA_LINKER_FLAGS
    ""
    CACHE STRING
    "Additional linker flags needed for the Cuda target")

  # KITSUNE FIXME: Find libdevice.bc
  set(KITSUNE_CUDA_LIBDEVICE_BC
    "")
endif()

if (KITSUNE_HIP_ENABLE)
  set(ROCM_PATH
    $ENV{ROCM_PATH}
    CACHE PATH
    "Path to hip install directory")

  # Probably should not be hard-coding /opt/rocm/* here, but it seems to be
  # where it goes in several distributions and it seems we cannot otherwise
  # reliably find rocm.
  find_package(hip REQUIRED
    PATHS $ENV{ROCM_PATH}/hip $ENV{ROCM_PATH}
    /opt/rocm/hip /opt/rocm)

  find_library(LIB_AMDHIP64 amdhip64 REQUIRED
    PATHS ${hip_LIBRARY_DIR}
    NO_DEFAULT_PATH)

  set(KITSUNE_HIP_DEFAULT_PREPROCESSOR_FLAGS
    "-D_tapir_hip_target")

  set(KITSUNE_HIP_DEFAULT_COMPILER_FLAGS
    "")

  set(KITSUNE_HIP_DEFAULT_LINKER_FLAGS
    "-L${ROCM_PATH}/lib -L${ROCM_PATH}/hip/lib -lamdhip64 -lpthread -lm -lrt -lkitrt")

  set(KITSUNE_HIP_EXTRA_PREPROCESSOR_FLAGS
    ""
    CACHE STRING
    "Additional preprocessor flags needed for the HIP target")

  set(KITSUNE_HIP_EXTRA_COMPILER_FLAGS
    ""
    CACHE STRING
    "Additional compiler flags needed for the HIP target")

  set(KITSUNE_HIP_EXTRA_LINKER_FLAGS
    ""
    CACHE STRING
    "Additional linker flags needed for the HIP target")
endif()

if (KITSUNE_OPENCILK_ENABLE)
  # We currently don't allow using a pre-built Cheetah. It would have to be
  # built with a "compatible" version of clang which is possible by using a
  # vanilla clang build of the same version, but it's probably not worth all the
  # error checking hassles that would be needed to detect if the versions were
  # mismatched.
  set(KITSUNE_CHEETAH_SOURCE_DIR ${KITSUNE_TARGETS_BINARY_DIR}/cheetah/cheetah)
  set(KITSUNE_CHEETAH_BINARY_DIR ${KITSUNE_TARGETS_BINARY_DIR}/cheetah/build)
  set(KITSUNE_CHEETAH_BUILD_ARGS
    -DCMAKE_C_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang
    -DCMAKE_CXX_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++
    -DCMAKE_ASM_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/lib/clang/${LLVM_VERSION_MAJOR}
    -DCHEETAH_DEFAULT_TARGET_TRIPLE=${LLVM_DEFAULT_TARGET_TRIPLE}
    -DCHEETAH_DIRECTORY_PER_TARGET=ON
  )

  ExternalProject_Add(cheetah ${KITSUNE_CHEETAH_SRC_DIR}
    DEPENDS        clang lld llvm-config llvm-link
    GIT_REPOSITORY https://github.com/OpenCilk/cheetah.git
    GIT_TAG        opencilk/v2.1
    SOURCE_DIR     ${KITSUNE_CHEETAH_SOURCE_DIR}
    BINARY_DIR     ${KITSUNE_CHEETAH_BINARY_DIR}
    CMAKE_ARGS     ${KITSUNE_CHEETAH_BUILD_ARGS})

  # The values here are populated given that we know where Cheetah will be
  # installed. The code to actual fetch, configure, build etc. is in
  # kitsune/runtime.
  set(KITSUNE_OPENCILK_DEFAULT_PREPROCESSOR_FLAGS
    "")

  set(KITSUNE_OPENCILK_DEFAULT_COMPILER_FLAGS
    "")

  # FIXME: We need to specify where libopencilk is installed with a -L flag?
  set(KITSUNE_OPENCILK_DEFAULT_LINKER_FLAGS
    "-lopencilk -lkitrt")

  set(KITSUNE_OPENCILK_EXTRA_PREPROCESSOR_FLAGS
    ""
    CACHE STRING
    "Additional preprocessor flags needed for the OpenCilk target")

  set(KITSUNE_OPENCILK_EXTRA_COMPILER_FLAGS
    ""
    CACHE STRING
    "Additional compiler flags needed for the OpenCilk target")

  set(KITSUNE_OPENCILK_EXTRA_LINKER_FLAGS
    ""
    CACHE STRING
    "Additional linker flags needed for the OpenCilk target")

  # Path to libopencilk-abi.bc. This will be generated when cheetah is built.
  set(KITSUNE_OPENCILK_LIBOPENCILKABI_BC
    "")

endif()

if (KITSUNE_OPENMP_ENABLE)
  # Do we need to check for libomp?

  set(KITSUNE_OPENMP_DEFAULT_PREPROCESSOR_FLAGS
    "")

  set(KITSUNE_OPENMP_DEFAULT_COMPILER_FLAGS
    "")

  set(KITSUNE_OPENMP_DEFAULT_LINKER_FLAGS
    "-fopenmp -lomp")

  set(KITSUNE_OPENMP_EXTRA_PREPROCESSOR_FLAGS
    ""
    CACHE STRING
    "Additional preprocessor flags needed for the OpenMP target_")

  set(KITSUNE_OPENMP_EXTRA_COMPILER_FLAGS
    ""
    CACHE STRING
    "Additional compiler flags needed for the OpenMP target")

  set(KITSUNE_OPENMP_EXTRA_LINKER_FLAGS
    ""
    CACHE STRING
    "Additional linker flags needed for the OpenMP target")
endif()

if (KITSUNE_QTHREADS_ENABLE)
  # FIXME: Do find_package or something here for qthreads. That will set
  # QTHREADS_INCLUDE_DIR and QTHREADS_LIBRARY_DIR below.
  message(FATAL_ERROR "Finding package for QThreads has not been implemented")

  # TODO: Do we really need the qthreads include directory to be in the search
  # path?
  set(KITSUNE_QTHREADS_DEFAULT_PREPROCESSOR_FLAGS
    "-I${QTHREADS_INCLUDE_DIR}")

  set(KITSUNE_QTHREADS_DEFAULT_COMPILER_FLAGS
    "")

  set(KITSUNE_QTHREADS_DEFAULT_LINKER_FLAGS
    "-L${QTHREADS_LIBRARY_DIR} -lqthread -lkitrt -lpthread -lhwloc")

  set(KITSUNE_QTHREADS_EXTRA_PREPROCESSOR_FLAGS
    ""
    CACHE STRING
    "Additional preprocessor flags needed for the Qthreads target")

  set(KITSUNE_QTHREADS_EXTRA_COMPILER_FLAGS
    ""
    CACHE STRING
    "Additional compiler flags needed for the Qthreads target")

  set(KITSUNE_QTHREADS_EXTRA_LINKER_FLAGS
    ""
    CACHE STRING
    "Additional linker flags needed for the Qthreads target")
endif()

if (KITSUNE_REALM_ENABLE)
  # FIXME: Do find_package or something here for realm. That will set
  # Realm_LIBRARY_DIR and other such variables which are used below.
  message(FATAL_ERROR "Finding package for Realm has not been implemented")

  set(KITSUNE_REALM_DEFAULT_PREPROCESSOR_FLAGS
    "")

  set(KITSUNE_REALM_DEFAULT_COMPILER_FLAGS
    "")

  set(KITSUNE_REALM_DEFAULT_LINKER_FLAGS
    "-L${Realm_LIBRARY_DIR} -L${Realm_WRAPPER_LIBRARY_DIR} -lrealm-abi -lrealm -lpthread -ldl -lrt -lkitrt")

  set(KITSUNE_REALM_EXTRA_PREPROCESSOR_FLAGS
    ""
    CACHE STRING
    "Additional preprocessor flags needed for the Realm target")

  set(KITSUNE_REALM_EXTRA_COMPILER_FLAGS
    ""
    CACHE STRING
    "Additional compiler flags needed for the Realm target")

  set(KITSUNE_REALM_EXTRA_LINKER_FLAGS
    ""
    CACHE STRING
    "Additional linker flags needed for the Realm target")
endif()

# FIXME KITSUNE: We probably don't need this. The -ftapir flag requires a
# backend and not providing it disables Tapir lowering altogether. This may be
# useful if we ever change that and stop requiring -ftapir. There is a
# reasonable argument to be made for why we should always lower using Tapir, but
# then this will have to be set to some runtime that is actually built.
set(KITSUNE_DEFAULT_TAPIR_RUNTIME "opencilk" CACHE STRING
  "Default Tapir runtime used by -ftapir.")

# This is ugly because it creates a dangling symlink because this is executed
# before clang is built. However, we force clang to be built, so we are sure
# of it not dangling by the time everything is built.
add_custom_target(${KITSUNE_C_FRONTEND} ALL
  ${CMAKE_COMMAND} -E create_symlink
  clang
  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${KITSUNE_C_FRONTEND})

add_custom_target(${KITSUNE_CXX_FRONTEND} ALL
  ${CMAKE_COMMAND} -E create_symlink
  clang++
  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${KITSUNE_CXX_FRONTEND})

foreach (link ${KITSUNE_C_FRONTEND} ${KITSUNE_CXX_FRONTEND})
  install(FILES
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${link}
    DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()

add_subdirectory(configs)
add_subdirectory(include/kitsune)

if(KITSUNE_INCLUDE_TESTS)
  add_subdirectory(test)
endif()

# We don't descend into the runtime subdirectory because that should only be
# configured when the runtimes are built. All the cmake environment variables
# that the runtime might need have to be passed through here.
#
# FIXME: Instead of passing all the Kitsune variables here by hand, look for
# for every variable starting with KITSUNE_, KITRT_ and LLVM_ and pass those
# in.
ExternalProject_Add(kitsune-runtimes
  DEPENDS llvm-config clang LLVM
  SOURCE_DIR ${KITSUNE_SOURCE_DIR}/runtime
  STAMP_DIR ${KITSUNE_BINARY_DIR}/runtime-stamp
  BINARY_DIR ${KITSUNE_BINARY_DIR}/runtime
  CMAKE_ARGS -DCMAKE_C_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang
             -DCMAKE_CXX_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
             -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
             -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
             -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
             -DKITSUNE_CUDA_ENABLE=${KITSUNE_CUDA_ENABLE}
             -DKITSUNE_HIP_ENABLE=${KITSUNE_HIP_ENABLE}
             -DKITSUNE_OPENCILK_ENABLE=${KITSUNE_OPENCILK_ENABLE}
             -DKITSUNE_OPENMP_ENABLE=${KITSUNE_OPENMP_ENABLE}
             -DKITSUNE_QTHREADS_ENABLE=${KITSUNE_QTHREADS_ENABLE}
             -DKITSUNE_REALM_ENABLE=${KITSUNE_REALM_ENABLE}
             -DKITSUNE_CUDA_VERSION_MIN=${KITSUNE_CUDA_VERSION_MIN}
             -DKITSUNE_ROMC_VERSION_MIN=${KITSUNE_ROCM_VERSION_MIN}
             -DKITRT_ENABLE_DEBUG=${KITRT_ENABLE_DEBUG}
             -DKITRT_ENABLE_VERBOSE=${KITRT_ENABLE_VERBOSE}
             -DKITCUDA_ENABLE_NVTX=${KITCUDA_ENABLE_NVTX}
             -DLLVM_CONFIG_PATH=${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-config
             -DLLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR}
             -DLLVM_VERSION_MINOR=${LLVM_VERSION_MINOR}
             -DLLVM_VERSION_PATCH=${LLVM_VERSION_PATCH}
             -DLLVM_INCLUDE_DIR=${LLVM_INCLUDE_DIR}
             -DLLVM_LIBRARY_DIR=${LLVM_LIBRARY_OUTPUT_INTDIR}
             -DLLVM_BINARY_DIR=${LLVM_RUNTIME_OUTPUT_INTDIR}
             -DLLVM_LIT_ARGS=${LLVM_LIT_ARGS}
             -DLLVM_LIBDIR_SUFFIX=${LLVM_LIBDIR_SUFFIX}
             -DLLVM_MAIN_INCLUDE_DIR=${LLVM_MAIN_INCLUDE_DIR}
             -DLLVM_RUNTIME_OUTPUT_INTDIR=${LLVM_RUNTIME_OUTPUT_INTDIR}
             -DCUDAToolkit_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}
             -DCUDAToolkit_LIBRARY_DIR=${CUDAToolkit_LIBRARY_DIR}
  INSTALL_COMMAND ""
  # STEP_TARGETS configure build
  USES_TERMINAL_CONFIGURE 1
  USES_TERMINAL_BUILD 1
  USES_TERMINAL_INSTALL 1)
