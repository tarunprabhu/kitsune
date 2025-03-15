#
# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune+tapir project.  It is released under
# the LLVM license.
#
cmake_minimum_required(VERSION 3.20)
project(kitsune C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(ExternalProject)
include(FetchContent)
include(GetClangResourceDir)
include(KitsuneUtils)

option(KITSUNE_INCLUDE_TESTS
  "Generate build targets for the Kitsune tests"
  ${LLVM_INCLUDE_TESTS})

option(KITSUNE_KOKKOS_ENABLED
  "Enable custom recognition and compilation of Kokkos"
  ON)

# FIXME: Should this be named KITRT_* since we don't have a separate kitcuda
# library any longer?
option(KITCUDA_ENABLE_NVTX
  "Enable NVTX profiling within the runtime"
  OFF)

# clang has deprecated GCC_INSTALL_PREFIX and --gcc-toolchain is also likely to
# be removed. The only way to use clang on systems where GCC is installed to a
# non-standard directory is to pass --gcc-install-dir on the command line. On
# HPC systems, spack packages install a config file that sets this argument so
# the user does not have to pass this explicitly on the command line. Gentoo
# does the same. Sometimes, we may not want to install a config file, in which
# case, the GCC_INSTALL_DIR can be provided at configure time. This is *NOT* a
# direct equivalent to clang's old GCC_INSTALL_DIR configure-time option. The
# argument passed here must be exactly that which would be passed to
# --gcc-install-dir. At runtime, if --gcc-install-dir is passed, that value will
# override this one.
set(KITSUNE_GCC_INSTALL_DIR "" CACHE STRING
  "The GCC installation directory to use. This should be the path to a
directory containing libgcc.a. This will nearly always be in the lib/
subdirectory in GCC's install directory. For instance, if GCC was installed to
/gcc_install, this is usually /gcc_install/lib/gcc/<triple>/<version> where
<triple> is the target triple for the system and <version> is the GCC version")

set(KITSUNE_C_FRONTEND kitcc)
set(KITSUNE_CXX_FRONTEND kit++)
set(KITSUNE_Fortran_FRONTEND kitfc)

# The name of Kitsune's main runtime library. This is created here because it is
# also used by the tests to ensure that check-kitsune does not run before the
# runtime has been built.
set(KITSUNE_LIBNAME kitrt)
set(KITSUNE_LIBNAME_STATIC "${KITSUNE_LIBNAME}_static")

# TODO: The _MAX variables are not currently used but it would be good to
# require a range instead of just a minimum. In practice, we may want to
# restrict it to just a single version because maintaining even a small range
# is likely to be a pain.
set(KITSUNE_CUDA_VERSION_MIN 12.3 CACHE INTERNAL "Minimum Cuda version required")
set(KITSUNE_CUDA_VERSION_MAX 12.3 CACHE INTERNAL "Maxmimum Cuda version supported")
set(KITSUNE_ROCM_VERSION_MIN 6.0 CACHE INTERNAL "Minimum ROCM version required")
set(KITSUNE_ROCM_VERSION_MAX 7.0 CACHE INTERNAL "Maximum ROCM version supported")

set(KITSUNE_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(KITSUNE_INCLUDE_DIR ${KITSUNE_SOURCE_DIR}/include)
set(KITSUNE_BINARY_DIR  ${PROJECT_BINARY_DIR})
set(KITSUNE_TARGETS_BINARY_DIR ${KITSUNE_BINARY_DIR}/targets)

# Just in case, check for utilities that are needed to build some of the
# dependencies. These are not needed for everything, but they are basic enough
# that we might as well check for them unconditionally and "globally".
find_program(GIT git REQUIRED)
find_program(GREP grep REQUIRED)
find_program(PATCH patch REQUIRED)
find_program(WC wc REQUIRED)

# Sanity check the value of KITSUNE_GCC_INSTALL_DIR if it was set
if (KITSUNE_GCC_INSTALL_DIR)
  find_library(LIBGCC gcc
    PATHS "${KITSUNE_GCC_INSTALL_DIR}"
    NO_DEFAULT_PATH)

  if (NOT LIBGCC)
    message(FATAL_ERROR "Could not find libgcc.a in "
      "'${KITSUNE_GCC_INSTALL_DIR}'. Check the value of "
      "KITSUNE_GCC_INSTALL_DIR")
  endif ()
endif ()

# This is just the subpath to the internal directory. It will be interpreted
# relative to some base path, typically ${CMAKE_INSTALL_PREFIX}
get_clang_resource_dir(CLANG_RESOURCE_SUBDIR)

# This is the clang resource directory within in the build directory.
get_clang_resource_dir(CLANG_RESOURCE_INTDIR
  PREFIX ${CMAKE_BINARY_DIR})

# This is the clang resource directory after installation.
get_clang_resource_dir(CLANG_RESOURCE_INSTALL_DIR
  PREFIX ${CMAKE_INSTALL_PREFIX})

if (KITSUNE_KOKKOS_ENABLED)
  # We currently don't support using a pre-built Kokkos - even if it has been
  # patched the way we need it to be. This could possibly be moved to
  # kitsune/runtime, just so everything that needs to be fetched and built is
  # there, even if this is not technically a runtime.

  set(KITSUNE_KOKKOS_BUILD_CONFIGURE_FLAGS ""
    CACHE STRING
    "Additional options to be passed to be CMake when configuring Kokkos.
    These are only used when building Kitsune")

  set(KITSUNE_KOKKOS_BUILD_COMPILE_FLAGS ""
    CACHE STRING
    "Additional C++ compiler flags needed to build Kokkos. These are only
     used when building Kitsune")

  set(KITSUNE_KOKKOS_SRC_DIR  ${KITSUNE_TARGETS_BINARY_DIR}/kokkos/kokkos)
  set(KITSUNE_KOKKOS_BUILD_DIR ${KITSUNE_TARGETS_BINARY_DIR}/kokkos/build)
  set(KITSUNE_KOKKOS_STAMP_DIR ${KITSUNE_TARGETS_BINARY_DIR}/kokkos/stamp)
  set(KITSUNE_KOKKOS_INSTALL_DIR ${KITSUNE_TARGETS_BINARY_DIR}/kokkos/install)
  set(KITSUNE_KOKKOS_BUILD_CMAKE_FLAGS
    -DCMAKE_CXX_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${KITSUNE_KOKKOS_INSTALL_DIR}
    -DCMAKE_CXX_FLAGS=${KITSUNE_KOKKOS_BUILD_COMPILE_FLAGS}
    -DKokkos_ENABLE_SERIAL=ON
    -DBUILD_TESTING=OFF
    -DBUILD_SHARED_LIBS=ON
    ${KITSUNE_KOKKOS_BUILD_CONFIGURE_FLAGS})

  # We have to patch kokkos to disable some sections of code that are using
  # exceptions.
  set(KITSUNE_KOKKOS_PATCH_EXCS ${KITSUNE_SOURCE_DIR}/patches/Kokkos_Exceptions.patch)

  # ----------------------------------------------------------------------------
  #                               WARNING
  #
  #   The PATCH_COMMAND below will succeed even if the patch fails to apply.
  #   Extra care must be taken when updating the GIT_TAG that the patch still
  #   applies cleanly.
  # ---------------------------------------------------------------------------
  #
  # The PATCH_COMMAND will only apply the patch if none of the files in the
  # checkout have been changed. This should be the case
  #
  #   - If this is the first time that Kokkos is being checked out
  #   - If the git tag was changed since the last time this was checked out
  #
  # The subcommands only use short options because the utilities on some systems
  # (such as the BSD's) do not support long options.
  #
  # TODO: Update Kokkos.
  #
  set(patch_warning "-- SKIPPING patch: Patch may have been applied already")
  ExternalProject_Add(kokkos
    DEPENDS                 clang lld llvm-config llvm-link
    GIT_REPOSITORY          https://github.com/kokkos/kokkos.git
    GIT_TAG                 606866d # 4.0.00
    SOURCE_DIR              ${KITSUNE_KOKKOS_SRC_DIR}
    BINARY_DIR              ${KITSUNE_KOKKOS_BUILD_DIR}
    STAMP_DIR               ${KITSUNE_KOKKOS_STAMP_DIR}
    CMAKE_ARGS              ${KITSUNE_KOKKOS_BUILD_CMAKE_FLAGS}
    PATCH_COMMAND           ${GIT} status --porcelain --untracked-files=no
                            | ${WC} -l
                            | ${GREP} -q 0
                            && ${PATCH} -p0 --input=${KITSUNE_KOKKOS_PATCH_EXCS}
                            || ${CMAKE_COMMAND} -E echo "${patch_warning}"
    USES_TERMINAL_PATCH     TRUE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD     TRUE
    USES_TERMINAL_INSTALL   TRUE
    UPDATE_DISCONNECTED     TRUE)

  # Copy the Kokkos files into a "staging" area. This is needed to run the
  # kitsune tests and also to allow kitsune to be run from the build directory
  # without requiring it to be installed.
  add_custom_target(kokkos-copy-to-build-dir ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${KITSUNE_KOKKOS_INSTALL_DIR}
    ${CLANG_RESOURCE_INTDIR}
    COMMENT "Copying Kokkos to the 'top-level' build directory"
    VERBATIM USES_TERMINAL
  )
  add_dependencies(kokkos-copy-to-build-dir kokkos)

  # Add a custom target to install Kokkos to the final install directory with
  # the rest of kitsune.
  #
  # NOTE: The trailing slash after the directory to copy is required
  #
  install(DIRECTORY ${KITSUNE_KOKKOS_INSTALL_DIR}/
    DESTINATION ${CLANG_RESOURCE_INSTALL_DIR}
    USE_SOURCE_PERMISSIONS)

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

if (KITSUNE_CUDA_ENABLED)
  message(STATUS "Kitsune: CUDA target enabled")

  find_package(CUDAToolkit ${KITSUNE_CUDA_VERSION_MIN} REQUIRED)

  # CUDAToolkit does not provide a CUDAToolkit_PREFIX variable. CUDAToolkit_ROOT
  # is not set either, even if the CUDAToolkit is found. The prefix relies on
  # environment variables, so prefer those. Otherwise, make a best guess at
  # figuring out what it is from the vriables that are set.
  #
  # Alternatively, we could require setting CUDA_PATH in the environment. That
  # is probably not necessary. On HPC machines (which generally give us the most
  # grief), CUDA_PATH is often set,so checking for it first should be ok. We
  # probably want to test this across a range of systems though.
  if (DEFINED $ENV{CUDA_PATH})
    message(STATUS "  Setting cuda prefix from CUDA_PATH environment variable")
    set(KISTUNE_CUDA_PREFIX $ENV{CUDA_PATH})
  elseif (DEFINED $ENV{CUDA_HOME})
    message(STATUS "  Setting cuda prefix from CUDA_HOME environment variable")
    set(KITSUNE_CUDA_PREFIX $ENV{CUDA_HOME})
  elseif (CUDAToolkit_ROOT)
    message(STATUS "  Setting cuda prefix from CUDAToolkit_ROOT")
    set(KITSUNE_CUDA_PREFIX ${CUDAToolkit_ROOT})
  else ()
    message(STATUS "  Setting cuda prefix to parent of CUDAToolkit_BIN_DIR")
    get_filename_component(KITSUNE_CUDA_PREFIX ${CUDAToolkit_BIN_DIR} DIRECTORY)
  endif ()
  if (NOT KITSUNE_CUDA_PREFIX)
    message(FATAL_ERROR "Could not determine cuda prefix")
  endif()

  set(KITSUNE_CUDA_VERSION_MAJOR ${CUDAToolkit_VERSION_MAJOR})
  set(KITSUNE_CUDA_VERSION_MINOR ${CUDAToolkit_VERSION_MINOR})
  set(KITSUNE_CUDA_VERSION_PATCH ${CUDAToolkit_VERSION_PATCH})
  set(KITSUNE_CUDA_VERSION_STRING "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}.${CUDAToolkit_VERSION_PATCH}")

  set(KITSUNE_CUDA_INCLUDE_DIR ${CUDAToolkit_INCLUDE_DIRS})
  set(KITSUNE_CUDA_BINARY_DIR ${CUDAToolkit_BIN_DIR})
  set(KITSUNE_CUDA_LIBRARY_DIR ${CUDAToolkit_LIBRARY_DIR})
  set(KITSUNE_CUDA_LIBDEVICE_DIR "${KITSUNE_CUDA_PREFIX}/nvvm/libdevice")

  message(STATUS "  Cuda prefix: ${KITSUNE_CUDA_PREFIX}")
  message(STATUS "  Cuda binary dir: ${KITSUNE_CUDA_BINARY_DIR}")
  message(STATUS "  Cuda include dir: ${KITSUNE_CUDA_INCLUDE_DIR}")
  message(STATUS "  Cuda library dir: ${KITSUNE_CUDA_LIBRARY_DIR}")

  set(KITSUNE_CUDA_LIBNAME_CUDART_STATIC cudart_static)
  set(KITSUNE_CUDA_LIBNAME_CUDART cudart)
  set(KITSUNE_CUDA_LIBNAME_CUDA cuda)

  # This should almost always be the libcuda that is installed on the system,
  # so look for it in the "usual" places.
  find_library(KITSUNE_CUDA_LIB_CUDA "${KITSUNE_CUDA_LIBNAME_CUDA}"  REQUIRED)

  get_filename_component(
    KITSUNE_CUDA_LIBCUDA_DIR "${KITSUNE_CUDA_LIB_CUDA}" DIRECTORY)

  find_library(KITSUNE_CUDA_LIB_CUDART "${KITSUNE_CUDA_LIBNAME_CUDART}" REQUIRED
    PATHS ${KITSUNE_CUDA_LIBRARY_DIR}
    NO_DEFAULT_PATH)

  find_library(KITSUNE_CUDA_LIB_CUDART_STATIC "${KITSUNE_CUDA_LIBNAME_CUDART_STATIC}" REQUIRED
    PATHS ${KITSUNE_CUDA_LIBRARY_DIR}
    NO_DEFAULT_PATH)

  find_library(KITSUNE_CUDA_LIB_NVPTX_STATIC nvptxcompiler_static REQUIRED
    PATHS ${KITSUNE_CUDA_LIBRARY_DIR}
    NO_DEFAULT_PATH)

  find_file(KITSUNE_CUDA_LIBDEVICE_BC libdevice.10.bc REQUIRED
    PATHS ${KITSUNE_CUDA_LIBDEVICE_DIR}
    NO_DEFAULT_PATH)

  find_program(KITSUNE_CUDA_PTXAS ptxas REQUIRED
    PATHS ${KITSUNE_CUDA_BINARY_DIR}
    NO_DEFAULT_PATH)

  find_program(KITSUNE_CUDA_FATBINARY fatbinary REQUIRED
    PATHS ${KITSUNE_CUDA_BINARY_DIR}
    NO_DEFAULT_PATH)

  set(KITSUNE_CUDA_ARCH_DEFAULT "sm_80"
    CACHE STRING
    "The default NVIDIA GPU architecture used by the cuda tapir target")

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
endif()

if (KITSUNE_HIP_ENABLED)
  message(STATUS "Kitsune: HIP target enabled.")

  # Apparently, ROCM_PATH needs to be set for Hip's build system to work
  # correctly. It is not clear if this is a problem with ROCM/HIP or a weird
  # interaction between that and Darwin. Whatever the case, we need this (pretty
  # lousy) workaround.
  if (NOT DEFINED ROCM_PATH)
    message(STATUS "  Kitsune HIP support enabled but ROCM_PATH not set.")
    if (DEFINED $ENV{ROCM_PATH})
      message(STATUS "  Using ROCM_PATH environment variable.")
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH
        "AMD ROCM/HIP installation directory.")
    else()
      message(STATUS "  Using the default install location [/opt/rocm].")
      set(ROCM_PATH "/opt/rocm" CACHE PATH
        "AMD ROCM/HIP installation directory.")
    endif()
  endif()
  message(STATUS "ROCm/HIP path: ${ROCM_PATH}")

  # The CMAKE_MODULE_PATH needs to be updated, probably because HIP doesn't know
  # how to find itself. Or something mad like that. I have given up trying to
  # write sane comments explaining all the workarounds to get AMD's software
  # stack working.
  list(APPEND CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake/hip")
  message(STATUS "   Updated cmake module search path: ${CMAKE_MODULE_PATH}")

  # Now we can finally look for the HIP package ...
  find_package(HIP MODULE REQUIRED)

  # ... but apparently we need to explicitly include the .cmake file otherwise
  # we don't get the variables that we need. WTH!
  include(${ROCM_PATH}/lib/cmake/hip/hip-config.cmake)

  # There ought to be a better way to set the HIP_PREFIX rather than simply
  # relying on ROCM_PATH, but if wishes were horses ...
  set(KITSUNE_HIP_PREFIX ${ROCM_PATH})
  set(KITSUNE_HIP_VERSION_MAJOR ${HIP_VERSION_MAJOR})
  set(KITSUNE_HIP_VERSION_MINOR ${HIP_VERSION_MINOR})
  set(KITSUNE_HIP_VERSION_PATCH ${HIP_VERSION_PATCH})
  set(KITSUNE_HIP_VERSION_STRING "${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}.${HIP_VERSION_PATCH}")
  set(KITSUNE_HIP_INCLUDE_DIR ${HIP_INCLUDE_DIR})
  set(KITSUNE_HIP_LIBRARY_DIR ${HIP_LIB_INSTALL_DIR})

  message(STATUS "  Hip prefix: ${KITSUNE_HIP_PREFIX}")
  message(STATUS "  Hip include dir: ${KITSUNE_HIP_INCLUDE_DIR}")
  message(STATUS "  Hip library dir: ${KITSUNE_HIP_LIBRARY_DIR}")

  set(KITSUNE_HIP_LIBNAME_AMDHIP "amdhip64")

  find_library(KITSUNE_HIP_LIB_AMDHIP64 ${KITSUNE_HIP_LIBNAME_AMDHIP} REQUIRED
    PATHS ${KITSUNE_HIP_LIBRARY_DIR} ${ROCM_PATH}/lib
    NO_DEFAULT_PATH)

  set(KITSUNE_HIP_ARCH_DEFAULT "gfx90a"
    CACHE STRING
    "The default AMD GPU architecture used by the hip tapir target")

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

  set(KITSUNE_HIP_BITCODE_DIR "${ROCM_PATH}/amdgcn/bitcode/")
  find_file(KITSUNE_HIP_BITCODE_FILE hip.bc REQUIRED
    PATHS ${KITSUNE_HIP_BITCODE_DIR}
    NO_DEFAULT_PATH)
  message(STATUS "hip bitcode file location: ${KITSUNE_HIP_BITCODE_DIR}")
endif()

if (KITSUNE_LAMBDA_ENABLED)
  message(STATUS "Kitsune: Lambda target enabled")

  message(WARNING "Lambda tapir target is experimental. Expect build failures
and other problems")

  set(KITSUNE_LAMBDA_EXTRA_PREPROCESSOR_FLAGS
    ""
    CACHE STRING
    "Additional preprocessor flags needed for the Lambda target")

  set(KITSUNE_LAMBDA_EXTRA_COMPILER_FLAGS
    ""
    CACHE STRING
    "Additional compiler flags needed for the Lambda target")

  set(KITSUNE_LAMBDA_EXTRA_LINKER_FLAGS
    ""
    CACHE STRING
    "Additional linker flags needed for the Lambda target")
endif ()

if (KITSUNE_OMPTASK_ENABLED)
  message(STATUS "Kitsune: OMPTask target enabled")

  message(WARNING "OMPTask tapir target is experimental. Expect build failures
and other problems")

  set(KITSUNE_OMPTASK_EXTRA_PREPROCESSOR_FLAGS
    ""
    CACHE STRING
    "Additional preprocessor flags needed for the OMPTask target")

  set(KITSUNE_OMPTASK_EXTRA_COMPILER_FLAGS
    ""
    CACHE STRING
    "Additional compiler flags needed for the OMPTask target")

  set(KITSUNE_OMPTASK_EXTRA_LINKER_FLAGS
    ""
    CACHE STRING
    "Additional linker flags needed for the OMPTask target")
endif ()

if (KITSUNE_OPENCILK_ENABLED)
  message(STATUS "Kitsune: OpenCilk target enabled")

  # We currently don't allow using a pre-built Cheetah. It would have to be
  # built with a "compatible" version of clang which is possible by using a
  # vanilla clang build of the same version, but it's probably not worth all the
  # error checking hassles that would be needed to detect if the versions were
  # mismatched.
  set(KITSUNE_OPENCILK_BUILD_CONFIGURE_FLAGS ""
    CACHE STRING
    "Additional options to be passed to be CMake when configuring Cheetah
    (the OpenCilk runtime). These are only used when building Kitsune")

  set(KITSUNE_OPENCILK_BUILD_COMPILE_FLAGS ""
    CACHE STRING
    "Additional C++ compiler flags needed to build Cheetah (the OpenCilk
    runtime). These are only used when building Kitsune")

  # We pass some LLVM_* variables to make Cheetah behave as if it were an
  # in-tree build.
  set(KITSUNE_CHEETAH_SOURCE_DIR ${KITSUNE_TARGETS_BINARY_DIR}/cheetah/cheetah)
  set(KITSUNE_CHEETAH_BINARY_DIR ${KITSUNE_TARGETS_BINARY_DIR}/cheetah/build)
  set(KITSUNE_CHEETAH_INSTALL_DIR ${KITSUNE_TARGETS_BINARY_DIR}/cheetah/install)
  set(KITSUNE_CHEETAH_BUILD_CMAKE_FLAGS
    -DCMAKE_C_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang
    -DCMAKE_CXX_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -DCMAKE_INSTALL_PREFIX=${KITSUNE_CHEETAH_INSTALL_DIR}
    -DCMAKE_CXX_FLAGS=${KITSUNE_OPENCILK_BUILD_COMPILE_FLAGS}
    -DCHEETAH_DEFAULT_TARGET_TRIPLE=${LLVM_DEFAULT_TARGET_TRIPLE}
    -DCHEETAH_DIRECTORY_PER_TARGET=ON
    -DLLVM_CMAKE_DIR=${CMAKE_BINARY_DIR}
    -DLLVM_COMMON_CMAKE_UTILS=${CMAKE_SOURCE_DIR}/llvm/cmake
    -DLLVM_LIBRARY_OUTPUT_INTDIR=${LLVM_LIBRARY_OUTPUT_INTDIR}
    -DLLVM_RUNTIME_OUTPUT_INTDIR=${LLVM_RUNTIME_OUTPUT_INTDIR}
    -DPACKAGE_VERSION=${PACKAGE_VERSION}
    ${KITSUNE_OPENCILK_BUILD_CONFIGURE_FLAGS})

  ExternalProject_Add(cheetah
    DEPENDS             clang lld llvm-config llvm-link
    GIT_REPOSITORY      https://github.com/OpenCilk/cheetah.git
    GIT_TAG             dev
    SOURCE_DIR          ${KITSUNE_CHEETAH_SOURCE_DIR}
    BINARY_DIR          ${KITSUNE_CHEETAH_BINARY_DIR}
    CMAKE_ARGS          ${KITSUNE_CHEETAH_BUILD_CMAKE_FLAGS}
    UPDATE_DISCONNECTED TRUE)

  # Copy the cheetah files into a "staging" area. This allows kitsune to be
  # run from the build directory. The destination directory is set to
  # ${CMAKE_BINARY_DIR} because Cheetah's build system is setup assuming that it
  # is being installed to the "top-level" of LLVM. In order for everything to
  # be copied correctly, we have to copy it to the "top-level" of the build
  # directory.
  add_custom_target(cheetah-copy-to-build-dir ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${KITSUNE_CHEETAH_INSTALL_DIR} ${CMAKE_BINARY_DIR}
    COMMENT "Copying OpenCilk to the 'top-level' build directory"
    VERBATIM USES_TERMINAL)
  add_dependencies(cheetah-copy-to-build-dir cheetah)

  # Add a custom target to install opencilk to the final install directory with
  # the rest of kitsune. The destination directory is set to ${CMAKE_BINARY_DIR}
  # since it is the install analog of ${CMAKE_BINARY_DIR}. See the comment
  # above the cheetah-copy-to-build-dir target for more details.
  #
  # NOTE: The trailing slash after the directory to copy is required.
  #
  install(DIRECTORY ${KITSUNE_CHEETAH_INSTALL_DIR}/
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    USE_SOURCE_PERMISSIONS)

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
endif()

if (KITSUNE_OPENMP_ENABLED)
  message(STATUS "Kitsune: OpenMP target enabled")

  # FIXME: Unconditionally fail so that when (if) the openmp tapir target is
  # ever resurrected, we know to look here and do whatever is appropriate.
  message(FATAL_ERROR "TODO: Do we need to check for libomp?")

  set(KITSUNE_OPENMP_EXTRA_PREPROCESSOR_FLAGS
    ""
    CACHE STRING
    "Additional preprocessor flags needed for the OpenMP target")

  set(KITSUNE_OPENMP_EXTRA_COMPILER_FLAGS
    ""
    CACHE STRING
    "Additional compiler flags needed for the OpenMP target")

  set(KITSUNE_OPENMP_EXTRA_LINKER_FLAGS
    ""
    CACHE STRING
    "Additional linker flags needed for the OpenMP target")
endif()

if (KITSUNE_QTHREADS_ENABLED)
  message(STATUS "Kitsune: Qthreads target enabled")

  # FIXME: Do find_package or ExternalProject_add or something else here for
  # qthreads. Odds are that we will end up building realm like we build Cheetah.
  # In any case, this message is here to ensure that when (if) we ever resurrect
  # the qthreads tapir target, we don't forget to do the right thing here.
  message(FATAL_ERROR "Finding/building QThreads has not been implemented")

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

if (KITSUNE_REALM_ENABLED)
  message(STATUS "Kitsune: Realm target enabled")

  # FIXME: Do find_package or ExternalProject_add or something else here for
  # realm. Odds are that we will end up building realm like we build Cheetah.
  # In any case, this message is here to ensure that when (if) we ever resurrect
  # the realm tapir target, we don't forget to do the right thing here.
  message(FATAL_ERROR "Finding/building Realm has not been implemented")

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

# Because we treat kitsune's runtime as an external project, we handle it the
# same way we do the tapir targets i.e., we create an "install" directory
# within the build directory into which all the files will be "installed" at
# build time with a separate install command to copy these out to the actual
# install directory for kitsune as specified by ${CMAKE_INSTALL_PREFIX}.
set(KITSUNE_RUNTIME_INSTALL_DIR ${KITSUNE_BINARY_DIR}/${KITSUNE_LIBNAME})

# -----------------------------------------------------------------------------
#
# The CLANG_, KITSUNE_ and LLVM_ variables that are passed in when configuring
# the runtime must be updated if any variables not already being passed are
# used within the runtime.
#
# NOTE: Instead of passing all the Kitsune variables here by hand, we could look
# for every variable starting with KITSUNE_, CLANG_ and LLVM_ and pass those in.
# Perhaps something like what is below.
#
#     get_cmake_property(vars VARIABLES)
#     list(FILTER vars INCLUDE REGEX "^KIT")
#     list(SORT vars)
#
# However, this may result in a number of "unused variable" warnings.
set(KITSUNE_RUNTIME_CONFIGURE_ARGS
  -DCMAKE_VERBOSE_BUILD=ON
  -DCMAKE_C_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang
  -DCMAKE_CXX_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
  -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
  -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
  -DCMAKE_INSTALL_PREFIX=${KITSUNE_RUNTIME_INSTALL_DIR}
  -DCMAKE_COLOR_DIAGNOSTICS=${CMAKE_COLOR_DIAGNOSTICS}
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
  -DCLANG_RESOURCE_INTDIR=${CLANG_RESOURCE_INTDIR}
  -DCLANG_RESOURCE_DIR=${CLANG_RESOURCE_INSTALL_DIR}
  -DCLANG_RESOURCE_SUBDIR=${CLANG_RESOURCE_SUBDIR}
  -DLLVM_CONFIG_PATH=${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-config
  -DLLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR}
  -DLLVM_VERSION_MINOR=${LLVM_VERSION_MINOR}
  -DLLVM_VERSION_PATCH=${LLVM_VERSION_PATCH}
  -DLLVM_BINARY_DIR=${LLVM_BINARY_DIR}
  -DLLVM_INCLUDE_DIR=${LLVM_INCLUDE_DIR}
  -DLLVM_LIBRARY_DIR=${LLVM_LIBRARY_DIR}
  -DLLVM_LIBRARY_OUTPUT_INTDIR=${LLVM_LIBRARY_OUTPUT_INTDIR}
  -DLLVM_RUNTIME_OUTPUT_INTDIR=${LLVM_RUNTIME_OUTPUT_INTDIR}
  -DLLVM_LIT_ARGS=${LLVM_LIT_ARGS}
  -DLLVM_LIBDIR_SUFFIX=${LLVM_LIBDIR_SUFFIX}
  -DLLVM_MAIN_INCLUDE_DIR=${LLVM_MAIN_INCLUDE_DIR}
  -DLLVM_RUNTIME_OUTPUT_INTDIR=${LLVM_RUNTIME_OUTPUT_INTDIR}
  -DKITSUNE_LIBNAME=${KITSUNE_LIBNAME}
  -DKITSUNE_LIBNAME_STATIC=${KITSUNE_LIBNAME_STATIC}
  -DKITSUNE_CUDA_ENABLED=${KITSUNE_CUDA_ENABLED}
  -DKITSUNE_HIP_ENABLED=${KITSUNE_HIP_ENABLED})
if (KITSUNE_CUDA_ENABLED)
  list(APPEND KITSUNE_RUNTIME_CONFIGURE_ARGS
    -DKITSUNE_CUDA_INCLUDE_DIR=${KITSUNE_CUDA_INCLUDE_DIR}
    -DKITSUNE_CUDA_LIBRARY_DIR=${KITSUNE_CUDA_LIBRARY_DIR}
    -DKITSUNE_CUDA_LIB_CUDA=${KITSUNE_CUDA_LIB_CUDA}
    -DKITSUNE_CUDA_LIB_CUDART=${KITSUNE_CUDA_LIB_CUDART}
    -DKITSUNE_CUDA_LIB_CUDART_STATIC=${KITSUNE_CUDA_LIB_CUDART_STATIC}
    -DKITSUNE_CUDA_LIB_NVPTX_STATIC=${KITSUNE_CUDA_LIB_NVPTX_STATIC}
    -DKITCUDA_ENABLE_NVTX=${KITCUDA_ENABLE_NVTX})
endif()
if (KITSUNE_HIP_ENABLED)
  list(APPEND KITSUNE_RUNTIME_CONFIGURE_ARGS
    -DKITSUNE_HIP_INCLUDE_DIR=${KITSUNE_HIP_INCLUDE_DIR}
    -DKITSUNE_HIP_LIBRARY_DIR=${KITSUNE_HIP_LIBRARY_DIR}
    -DKITSUNE_HIP_LIB_AMDHIP64=${KITSUNE_HIP_LIB_AMDHIP64})
endif ()

# We don't descend into the runtime subdirectory because that should only be
# configured when the runtimes are built. All the cmake environment variables
# that the runtime might need have to be passed through here. BUILD_ALWAYS must
# be set to TRUE here, otherwise runtime/ will not be checked for changes, which
# is not terribly helpful for Kitsune developers.
ExternalProject_Add(${KITSUNE_LIBNAME}
  DEPENDS                    llvm-config clang LLVM
  SOURCE_DIR                 ${KITSUNE_SOURCE_DIR}/runtime
  STAMP_DIR                  ${KITSUNE_BINARY_DIR}/stamp
  BINARY_DIR                 ${KITSUNE_BINARY_DIR}/runtime
  CMAKE_ARGS                 ${KITSUNE_RUNTIME_CONFIGURE_ARGS}
  USES_TERMINAL_CONFIGURE    TRUE
  USES_TERMINAL_BUILD        TRUE
  USES_TERMINAL_INSTALL      TRUE
  BUILD_ALWAYS               TRUE
  CONFIGURE_HANDLED_BY_BUILD TRUE)

add_subdirectory(configs)
add_subdirectory(include/kitsune)
add_subdirectory(tools)
add_subdirectory(experiments)

if(KITSUNE_INCLUDE_TESTS)
  add_subdirectory(test)
endif()

# Create the rules to create a "base" Kitsune executable. This is simply a copy
# of the base clang (or flang) executable. We have to create a copy because a
# symlink will not work as we want. Very early in the process, the symlinks are
# stripped away and the underlying executable is exposed to the driver. We need
# this to *not* be named clang in order to determine whether we should be
# compiling in "kitsune mode" (this changes the default behavior for certain
# things such as disabling exceptions, setting the fp-contract option to a
# different value etc. so we have to get this right). To be consistent with
# clang, we do not create a separate frontend for C++.
set(kitcc_main_file "${KITSUNE_C_FRONTEND}-${LLVM_VERSION_MAJOR}")
set(kitcc_main "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${kitcc_main_file}")
add_custom_target(kitcc-main ALL
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:clang> ${kitcc_main}
  COMMENT "Creating kitcc executable"
  VERBATIM USES_TERMINAL)
add_dependencies(kitcc-main clang)
install(PROGRAMS
  "${kitcc_main}"
  DESTINATION ${CMAKE_INSTALL_BINDIR})
setup_frontend_symlink(${KITSUNE_C_FRONTEND} ${kitcc_main_file} kitcc-main)
setup_frontend_symlink(${KITSUNE_CXX_FRONTEND} ${kitcc_main_file} kitcc-main)

if (KITSUNE_Fortran_ENABLED)
  set(kitfc_main_file "${KITSUNE_Fortran_FRONTEND}-${LLVM_VERSION_MAJOR}")
  set(kitfc_main "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${kitfc_main_file}")
  add_custom_target(kitfc-main ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:flang-new> "${kitfc_main}"
    COMMENT "Creating kitfc executable"
    VERBATIM USES_TERMINAL)
  add_dependencies(kitfc-main flang-new)
  install(PROGRAMS
    "${kitfc_main}"
    DESTINATION ${CMAKE_INSTALL_BINDIR})
  setup_frontend_symlink(${KITSUNE_Fortran_FRONTEND} ${kitfc_main_file} kitfc-main)
endif()

# Copy the kitsune runtime files into a "staging" area. This allows kitsune to
# be run from the build directory.
add_custom_target(kitrt-copy-to-build-dir ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${KITSUNE_RUNTIME_INSTALL_DIR} ${CMAKE_BINARY_DIR}
  COMMENT "Copying kitrt to the 'top-level' build directory"
  VERBATIM USES_TERMINAL
)
add_dependencies(kitrt-copy-to-build-dir ${KITSUNE_LIBNAME})

# NOTE: The trailing slash after the directory to copy is required.
install(DIRECTORY ${KITSUNE_RUNTIME_INSTALL_DIR}
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  USE_SOURCE_PERMISSIONS)
