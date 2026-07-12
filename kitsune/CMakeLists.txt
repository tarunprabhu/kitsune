#
# Copyright (c) 2021 Triad National Security, LLC. All rights reserved.
#
# This file is part of the kitsune project. It is released under the LLVM
# license.
#

cmake_minimum_required(VERSION 3.20)
project(kitsune C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(ExternalProject)
include(FetchContent)
include(GetClangResourceDir)

# ----------------------------------- NOTES -----------------------------------
#
# Kitsune's runtime dependencies *MUST* be built here because some CMake
# variables that are set while building those dependencies are used in
# subdirectories other than runtime/, such as include/ and test/.
#
# The dependencies that are built together with Kitsune i.e. Cheetah, Kokkos
# etc. (but not cuda and hip) are always installed to ${KITSUNE_RESOURCE_DIR}.
# This is ${CMAKE_INSTALL_PREFIX}/lib<suffix>/kitsune/${LLVM_VERSION_MAJOR}.
# This makes these dependencies "internal" to Kitsune. This is intentional. We
# currently do not support using an external build of these dependencies since
# Kitsune will only work with specific versions of the dependencies (we have no
# intention of supporting a range of versions simply because of the software
# engineering effort that entails). If we were to install these to
# ${CMAKE_INSTALL_PREFIX}, we may conflict with versions of these libraries that
# may be installed there.
#
# While most of these dependencies are built with the compiler used to build
# Kitsune, some, like Cheetah, must be built with Kitsune itself. When doing so,
# we suppress the checks that the compiler works by setting
# `-DCMAKE_<LANG>_COMPILER_WORKS=1` in the Cheetah's configuration command. On
# On systems where clang requires --sysroot to function correctly (such as many
# HPC clusters with their wonderfully antique system software stack), the
# compiler checks often fail since they do not set --sysroot. It is reasonable
# to expect that the clang that was just build works (if it doesn't, the cause
# should nearly always be a bug introduced by a Kitsune developer who will,
# naturally, fix the issue promptly). We ensure that
# --sysroot=${KITSUNE_SYSROOT} is added to the build commands for such
# dependencies, so, barring other issues, the build should succeed.
#
# TODO: A better fix might be to generate a configuration file in this case just
# for the purpose of building the tool/runtime with clang with --sysroot set.
# But this might result in other issues related to the tests. It would be
# worth exploring this option though.
#
# ------------------------------------------------------------------------------

# Prepare the variable OUT for use in an ExternalProject command. INP is a list
# of compiler/linker options. All elements of INP will be copied to OUT. In
# addition to this, the --sysroot= option will be added to OUT if
# KITSUNE_SYSROOT is not empty. OUT will use '|' as the list separator instead
# of the usual ';'. Because ExternalProject issues commands on a terminal, the
# use of ';' is unreliable since the terminal may interpret it as an
# end-of-command marker. '|' is safer since it is unlikely to appear in library,
# or path, names.
macro(prepare_extproj_cmdline_opts INP OUT)
  set(${OUT} "${INP}")
  if (KITSUNE_SYSROOT)
    string(APPEND ${OUT} " --sysroot=${KITSUNE_SYSROOT}")
  endif ()
  string(REPLACE ";" "|" ${OUT} "${${OUT}}")
endmacro ()

# FIXME: Should this be named KITRT_* since we don't have a separate kitcuda
# library any longer?
# FIXME: The support for this is probably badly broken since all the functions
# have since become part of a header-only library, so even the code in Kitsune's
# runtime likely needs fixing.
option(KITCUDA_ENABLE_NVTX
  "Enable NVTX profiling within the runtime"
  OFF)

# Convenience variables that are shorter.
set(SHLIB_PFX ${CMAKE_SHARED_LIBRARY_PREFIX})
set(SHLIB_SFX ${CMAKE_SHARED_LIBRARY_SUFFIX})
set(ARCHIVE_PFX ${CMAKE_STATIC_LIBRARY_PREFIX})
set(ARCHIVE_SFX ${CMAKE_STATIC_LIBRARY_SUFFIX})

set(KITSUNE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KITSUNE_INCLUDE_DIR ${KITSUNE_SOURCE_DIR}/include)
set(KITSUNE_BINARY_DIR ${PROJECT_BINARY_DIR})
set(KITSUNE_EXTDEPS_BINARY_DIR ${KITSUNE_BINARY_DIR}/extdeps)

set(KITRT_SOURCE_DIR ${KITSUNE_SOURCE_DIR}/runtime)
set(KITRT_BINARY_DIR ${KITSUNE_BINARY_DIR}/runtime)

# Just in case, check for utilities that are needed to build some of the
# dependencies. These are not needed for everything, but they are basic enough
# that we might as well check for them unconditionally and "globally".
find_program(GIT git REQUIRED)
find_program(GREP grep REQUIRED)
find_program(PATCH patch REQUIRED)
find_program(WC wc REQUIRED)

find_library(LIB_DL dl REQUIRED)
list(APPEND KITSUNE_COMMON_LIBS ${LIB_DL})

# The pthread functions are now part of libc on Linux and FreeBSD, and they
# have always been part of libSystem on MacOSX. libpthread was removed from
# glibc in version 2.34. There is no need to explicitly link this in unless we
# have older versions of glibc around. We should consider removing this from
# here at some point when we are certain we don't need this. That might be a
# while since HPC machines are notorious for having antiquated software stacks.
#
# https://developers.redhat.com/articles/2021/12/17/why-glibc-234-removed-libpthread
#
find_library(LIB_PTHREAD pthread REQUIRED)
list(APPEND KITSUNE_COMMON_LIBS ${LIB_PTHREAD})

# The librt functions are part of glibc as of v2.34. There is no need to link
# this in explicitly unless we have older versions of libc around. We do have to
# suppport HPC clusters with old libc installations, so this might have to be
# around for some time yet.
find_library(LIB_RT rt REQUIRED)
list(APPEND KITSUNE_COMMON_LIBS ${LIB_RT})

# On FreeBSD, the backtrace-related functions are defined in libexecinfo. On
# other systems, they are part of libc.
if (CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
  find_library(LIB_EXECINFO execinfo REQUIRED)
  list(APPEND KITSUNE_COMMON_LIBS ${LIB_EXECINFO})
endif()

# Clang resource directory in the build directory
get_clang_resource_dir(CLANG_RESOURCE_INTDIR
  PREFIX ${CMAKE_BINARY_DIR})

# Resource directory in the build directory
set(KITSUNE_RESOURCE_INTDIR
  ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/kitsune/${LLVM_VERSION_MAJOR})

# Resource directory in the install directory
set(KITSUNE_RESOURCE_DIR
  ${CMAKE_INSTALL_PREFIX}/lib${LLVM_LIBDIR_SUFFIX}/kitsune/${LLVM_VERSION_MAJOR})

if (KITSUNE_PAPI_ENABLED)
  # We currently don't allow using a pre-built PAPI. It is not readily available
  # in the package managers of the major Linux distributions, so it may be
  # easier to just build it ourselves.
  message(STATUS "Kitsune: PAPI support enabled")

  # TODO: This has not been tested anywhere other than Linux. See if this can be
  # enabled on other platforms.
  if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "PAPI support in Kitsune is not available on '${CMAKE_SYSTEM_NAME}'")
  endif ()

  # The configure script that is generated by PAPI contains bash-specific
  # syntax. This should be available on most platforms that we care about.
  find_program(BASH bash REQUIRED)

  # For now, PAPI support in Kitsune is only available on Linux. In most cases,
  # the `make` that is found will be GNU make. On other systems, this might not
  # be the case. For instance, on FreeBSD, `make` is usually BSD make. The
  # flavor of make will change the command-line options.
  #
  # When we support PAPI on other platforms, we should ensure that we always use
  # GNU make since it is convenient to just build PAPI with `make -j`.
  find_program(MAKE make REQUIRED)

  set(LIBPAPI_NAME ${ARCHIVE_PFX}papi${ARCHIVE_SFX})
  set(LIBPAPI ${KITSUNE_RESOURCE_INTDIR}/lib/${LIBPAPI_NAME})

  # PAPI-7.2.0 fails to build if -std=c23 is used. This is the default since
  # GCC 15, so set the standard explicitly. It is not clear if PAPI builds with
  # -fPIC automatically, so add it explicitly.
  set(CFLAGS "-std=gnu17 -fPIC")

  set(KITSUNE_PAPI_CONFIGURE_FLAGS
    --disable-fortran
    --with-shared-lib=no
    --with-static-lib=yes
    --prefix=${KITSUNE_RESOURCE_INTDIR})
  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND KITSUNE_PAPI_CONFIGURE_FLAGS --with-perf-events)
  endif ()

  # Notes about the commands below:
  #
  #   - The configure script must be run from the src/ directory within PAPI.
  #     There might be a way to get `make` to do the right thing if it is run
  #     from elsewhere, but if there is, I haven't found it.
  #
  #   - BUILD_COMMAND assumes that `make` is GNU make since it uses `-j`
  #     without a value, as is required by, for instance, BSD make. Since this
  #     is currently only supported on Linux, it is ok to expect that we will
  #     always have GNU make.
  #
  #   - The quoting of the CFLAGS environment variable is the only one known to
  #     work when more than one flag is to be passed. I have no idea why.
  #
  # The GIT_TAG must be updated when we wish to use a newer version of PAPI.
  ExternalProject_Add(papi
    GIT_REPOSITORY          https://github.com/icl-utk-edu/papi
    GIT_TAG                 papi-7-2-0-t
    PREFIX                  ${KITSUNE_EXTDEPS_BINARY_DIR}/papi
    INSTALL_DIR             ${KITSUNE_RESOURCE_INTDIR}
    CONFIGURE_COMMAND       cd src &&
                            ${CMAKE_COMMAND} -E env
                            CC=${CMAKE_C_COMPILER}
                            "CFLAGS=${CFLAGS}"
                            CONFIG_SHELL=${BASH}
                            --
                            ./configure ${KITSUNE_PAPI_CONFIGURE_FLAGS}
    BUILD_COMMAND           ${MAKE} -C src -j
    INSTALL_COMMAND         ${MAKE} -C src install
    BUILD_IN_SOURCE         TRUE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD     TRUE
    USES_TERMINAL_INSTALL   TRUE
    UPDATE_DISCONNECTED     TRUE
    EXCLUDE_FROM_ALL        TRUE
    INSTALL_BYPRODUCTS      ${LIBPAPI})

  list(APPEND KITRT_DEPENDS papi)

  set(KITSUNE_PAPI_INCDIRS ${KITSUNE_RESOURCE_INTDIR}/include)
  set(KITSUNE_PAPI_LIBS ${LIBPAPI})
endif ()

if (KITSUNE_KOKKOS_ENABLED)
  # We currently don't support using a pre-built Kokkos - even if it has been
  # patched the way we need it to be. This could possibly be moved to
  # kitsune/runtime, just so everything that needs to be fetched and built is
  # there, even if this is not technically a runtime.
  message(STATUS "Kitsune: Kokkos-mode enabled")

  set(KITSUNE_KOKKOS_BUILD_CONFIGURE_FLAGS ""
    CACHE STRING
    "Additional options to be passed to be CMake when configuring Kokkos. These
    are only used when building Kokkos as part of building Kitsune itself")

  set(KITSUNE_KOKKOS_BUILD_COMPILE_FLAGS ""
    CACHE STRING
    "Additional C++ compiler flags needed to build Kokkos. These are only used
     when building Kokkos as part of building Kitsune itself")

  set(KITSUNE_KOKKOS_BUILD_LINK_FLAGS ""
    CACHE STRING
    "Additional linker options needed to build Kokkos. These are only used when
     building Kokkos as part of building Kitsune itself")

  prepare_extproj_cmdline_opts("${KITSUNE_KOKKOS_BUILD_COMPILE_FLAGS}"
    KITSUNE_KOKKOS_CXX_FLAGS)
  prepare_extproj_cmdline_opts("${KITSUNE_KOKKOS_BUILD_LINK_FLAGS}"
    KITSUNE_KOKKOS_LINKER_FLAGS)

  set(KITSUNE_KOKKOS_BASE_DIR ${KITSUNE_EXTDEPS_BINARY_DIR}/kokkos)
  set(KITSUNE_KOKKOS_SOURCE_DIR ${KITSUNE_KOKKOS_BASE_DIR}/kokkos)
  set(KITSUNE_KOKKOS_BINARY_DIR ${KITSUNE_KOKKOS_BASE_DIR}/build)
  set(KITSUNE_KOKKOS_INSTALL_DIR ${KITSUNE_KOKKOS_BASE_DIR}/install)

  # These are the only headers and libraries used in Kitsune's Kokkos-mode demos
  # and tests.
  set(LIBKOKKOS_NAME ${SHLIB_PFX}kokkoscore${SHLIB_SFX})
  set(LIBKOKKOS ${KITSUNE_RESOURCE_INTDIR}/kokkos/lib/${LIBKOKKOS_NAME})
  set(KOKKOS_H ${KITSUNE_RESOURCE_INTDIR}/kokkos/include/Kokkos_Core.hpp)

  # Here, we set Kokkos_ENABLE_SERIAL to ON, and don't build any parallel
  # "backends". This is intentional. We build Kokkos ourselves because we have
  # to patch it to remove exceptions. It isn't used for anything except for
  # testing Kitsune's Kokkos-mode - and even that is just a demonstration.
  set(KITSUNE_KOKKOS_CMAKE_FLAGS
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_CXX_FLAGS=${KITSUNE_KOKKOS_CXX_FLAGS}
    -DCMAKE_EXE_LINKER_FLAGS=${KITSUNE_KOKKOS_LINKER_FLAGS}
    -DCMAKE_SHARED_LINKER_FLAGS=${KITSUNE_KOKKOS_LINKER_FLAGS}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_COLOR_DIAGNOSTICS=${CMAKE_COLOR_DIAGNOSTICS}
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -DCMAKE_INSTALL_PREFIX=${KITSUNE_RESOURCE_INTDIR}/kokkos
    -DCMAKE_INSTALL_LIBDIR=lib
    -DKokkos_ENABLE_SERIAL=ON
    -DBUILD_TESTING=OFF
    -DBUILD_SHARED_LIBS=ON
    ${KITSUNE_KOKKOS_BUILD_CONFIGURE_FLAGS})

  ExternalProject_Add(kokkos
    GIT_REPOSITORY          https://github.com/kokkos/kokkos.git
    GIT_TAG                 4.7.00
    PREFIX                  ${KITSUNE_EXTDEPS_BINARY_DIR}/kokkos
    INSTALL_DIR             ${KITSUNE_RESOURCE_INTDIR}/kokkos
    CMAKE_ARGS              ${KITSUNE_KOKKOS_CMAKE_FLAGS}
    LIST_SEPARATOR          |
    USES_TERMINAL_PATCH     TRUE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD     TRUE
    USES_TERMINAL_INSTALL   TRUE
    UPDATE_DISCONNECTED     TRUE
    EXCLUDE_FROM_ALL        TRUE)

  # When Kitsune is installed, we need to make sure Kokkos is installed as well.
  #
  # We copy the headers and libraries into a dedicated kokkos subdirectory to
  # avoid polluting the directories containing the other headers and libraries
  # in clang's resource directory. Making sure that the headers are included
  # requires additional support in the compiler to make sure that
  # `#include<Kokkos_Core.hpp>` works. But, we have special support for Kokkos
  # anyway, so this is reasonable.
  #
  # NOTE: The trailing slash after the directory to copy is required
  install(DIRECTORY ${KITSUNE_RESOURCE_INTDIR}/kokkos/include/
    DESTINATION ${KITSUNE_RESOURCE_DIR}/kokkos/include/
    USE_SOURCE_PERMISSIONS)
  install(DIRECTORY ${KITSUNE_RESOURCE_INTDIR}/kokkos/lib/
    DESTINATION ${KITSUNE_RESOURCE_DIR}/kokkos/lib
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING
    PATTERN "${SHLIB_PFX}kokkoscore.*"
    PATTERN "cmake" EXCLUDE)

  # The runtime does not actually depend on this, but we need some way to run
  # the target since it is excluded from ALL.
  list(APPEND KITRT_DEPENDS kokkos)
endif()

set(KITSUNE_CUDA_VERSION_MAJOR 0)
set(KITSUNE_CUDA_VERSION_MINOR 0)
set(KITSUNE_CUDA_VERSION_PATCH 0)
if (KITSUNE_CUDA_ENABLED)
  message(STATUS "Kitsune: cuda tapir target enabled")

  set(KITSUNE_CUDA_ARCH_DEFAULT "sm_80"
    CACHE STRING
    "The default NVIDIA GPU architecture used by the cuda tapir target when a GPU is not detected on the system")

  # We cannot request a version range in the call to find_package below because
  # CUDAToolkit does not support version ranges. Instead, we check the version
  # that was found and raise an error if it is not in the supported range.
  #
  # CUDAToolkit_ROOT will already have been set correctly in llvm/CMakeLists.txt
  find_package(CUDAToolkit REQUIRED)

  # FIXME: We do not test for all versions in this range. In all likelihood,
  # only some of the versions in the range below actually work. We should setup
  # some infrastructure that actually tests these so we can keep a known
  # working range.

  # The minimum cuda version that is supported
  set(KITSUNE_CUDA_VERSION_MIN 12.8)

  # The lowest unsupported cuda version that is greater than the minimum
  # supported version
  set(KITSUNE_CUDA_VERSION_MAX 13.4)

  set(KITSUNE_CUDA_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR})
  if (KITSUNE_CUDA_VERSION VERSION_LESS KITSUNE_CUDA_VERSION_MIN
      OR KITSUNE_CUDA_VERSION VERSION_GREATER_EQUAL KITSUNE_CUDA_VERSION_MAX)
    message(FATAL_ERROR
      "cuda version '${KITSUNE_CUDA_VERSION}' not in supported range ${KITSUNE_CUDA_VERSION_MIN}...<${KITSUNE_CUDA_VERSION_MAX}")
  endif ()

  set(KITSUNE_CUDA_VERSION_MAJOR ${CUDAToolkit_VERSION_MAJOR})
  set(KITSUNE_CUDA_VERSION_MINOR ${CUDAToolkit_VERSION_MINOR})
  set(KITSUNE_CUDA_VERSION_PATCH ${CUDAToolkit_VERSION_PATCH})

  # CUDAToolkit does not provide a CUDAToolkit_PREFIX variable. CUDAToolkit_ROOT
  # is not set either, even if the CUDAToolkit is found. The prefix relies on
  # environment variables, so prefer those. Otherwise, make a best guess at
  # figuring out what it is from the variables that are set.
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

  message(STATUS "  Cuda prefix: ${KITSUNE_CUDA_PREFIX}")
  message(STATUS "  Cuda binary dirs: ${CUDAToolkit_BIN_DIR}")
  message(STATUS "  Cuda include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
  message(STATUS "  Cuda library dirs: ${CUDAToolkit_LIBRARY_DIR}")

  list(APPEND KITSUNE_CUDA_INCDIRS "${CUDAToolkit_INCLUDE_DIRS}")

  # This is expected to be installed on the system, so look for it in the usual
  # places.
  find_library(KITSUNE_CUDA_LIB_CUDA cuda REQUIRED)
  list(APPEND KITSUNE_CUDA_LIBS ${KITSUNE_CUDA_LIB_CUDA})

  find_library(KITSUNE_CUDA_LIB_CUDART_STATIC cudart_static REQUIRED
    PATHS ${CUDAToolkit_LIBRARY_DIR}
    NO_DEFAULT_PATH)
  list(APPEND KITSUNE_CUDA_LIBS ${KITSUNE_CUDA_LIB_CUDART_STATIC})

  find_program(KITSUNE_CUDA_PTXAS ptxas REQUIRED
    PATHS ${CUDAToolkit_BIN_DIR}
    NO_DEFAULT_PATH)

  find_program(KITSUNE_CUDA_FATBINARY fatbinary REQUIRED
    PATHS ${CUDAToolkit_BIN_DIR}
    NO_DEFAULT_PATH)

  if (KITCUDA_ENABLE_NVTX)
    list(APPEND KITSUNE_CUDA_DEFNS KITCUDA_ENABLE_NVTX)
  endif()
endif()

# We always set `KITSUNE_CUDA_VERSION_STRING` because it is used in Kitsune's
# build configuration object. It is ok if it is simply "0.0".
set(KITSUNE_CUDA_VERSION_STRING
  "${KITSUNE_CUDA_VERSION_MAJOR}.${KITSUNE_CUDA_VERSION_MINOR}")

set(KITSUNE_HIP_VERSION_MAJOR 0)
set(KITSUNE_HIP_VERSION_MINOR 0)
set(KITSUNE_HIP_VERSION_PATCH 0)
if (KITSUNE_HIP_ENABLED)
  message(STATUS "Kitsune: hip tapir target enabled.")

  set(KITSUNE_HIP_ARCH_DEFAULT "gfx90a"
    CACHE STRING
    "The default AMD GPU architecture used by the hip tapir target when a GPU is not detected on the system")

  # We need to add both lib and lib64 because some systems install the hip
  # libraries to lib64 while others install it in lib. ROCM_PATH will already
  # have been set correctly in llvm/CMakeLists.txt
  #
  # We cannot check for the range of versions here because cmake's version
  # constraints do not work as expected when the major versions of the minimum
  # and maximum supported versions are different. Instead, we check the version
  # of the hip package that is found.
  find_package(HIP REQUIRED
    PATHS ${ROCM_PATH}/lib/cmake/hip ${ROCM_PATH}/lib64/cmake/hip
    NO_DEFAULT_PATH)

  # FIXME: We do not test for all versions in this range. In all likelihood,
  # only some of the versions in the range below actually work. We should setup
  # some infrastructure that actually tests these so we can keep a known
  # working range.

  # The minimum ROCm version that is supported
  set(KITSUNE_HIP_VERSION_MIN 6.3)

  # The lowest unsupported ROCm version that is greater than the minimum
  # supported version
  set(KITSUNE_HIP_VERSION_MAX 7.3)

  set(KITSUNE_HIP_VERSION
    ${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}.${HIP_VERSION_PATCH})
  if (KITSUNE_HIP_VERSION VERSION_LESS KITSUNE_HIP_VERSION_MIN
      OR KITSUNE_HIP_VERSION VERSION_GREATER_EQUAL KITSUNE_HIP_VERSION_MAX)
    message(FATAL_ERROR
      "ROCm version '${KITSUNE_HIP_VERSION}' not in supported range ${KITSUNE_HIP_VERSION_MIN}...<${KITSUNE_HIP_VERSION_MAX}")
  endif ()

  set(KITSUNE_HIP_VERSION_MAJOR ${HIP_VERSION_MAJOR})
  set(KITSUNE_HIP_VERSION_MINOR ${HIP_VERSION_MINOR})
  set(KITSUNE_HIP_VERSION_PATCH ${HIP_VERSION_PATCH})

  # There ought to be a better way to set the HIP_PREFIX rather than simply
  # relying on ROCM_PATH, but if wishes were horses.... In all code following
  # this, we should use KITSUNE_HIP_PREFIX instead of ROCM_PATH for consistency.
  set(KITSUNE_HIP_PREFIX ${ROCM_PATH})

  message(STATUS "  Hip prefix: ${KITSUNE_HIP_PREFIX}")
  message(STATUS "  Hip include dirs: ${HIP_INCLUDE_DIR}")
  message(STATUS "  Hip library dirs: ${HIP_LIB_INSTALL_DIR}")

  list(APPEND KITSUNE_HIP_INCDIRS "${HIP_INCLUDE_DIR}")

  find_library(KITSUNE_HIP_LIB_AMDHIP64 amdhip64 REQUIRED
    PATHS ${HIP_LIB_INSTALL_DIR}
    NO_DEFAULT_PATH)
  list(APPEND KITSUNE_HIP_LIBS ${KITSUNE_HIP_LIB_AMDHIP64})

  # We need to find the directory containing the bitcode files. Depending on the
  # system, this could be in one of several places. ${HIP_PREFIX}/amdgcn/bitcode
  # is a legacy location, but look there anyway because, on some HPC platforms
  # at least, the files can still be found there. We look for hip.bc because,
  # as of ROCm-6.3 at least, that was still guaranteed to exist. But if that
  # ever changes, we may need to look for a different bitcode file here.
  find_file(HIP_BC hip.bc REQUIRED
    PATHS ${KITSUNE_HIP_PREFIX}/amdgcn/bitcode
    ${KITSUNE_HIP_PREFIX}/lib/amdgcn/bitcode
    ${KITSUNE_HIP_PREFIX}/lib64/amdgcn/bitcode
    NO_DEFAULT_PATH)
  get_filename_component(KITSUNE_HIP_BITCODE_DIR "${HIP_BC}" DIRECTORY)
  message(STATUS "hip bitcode file location: ${KITSUNE_HIP_BITCODE_DIR}")
endif()

# We always set `KITSUNE_HIP_VERSION_STRING` because it is used in Kitsune's
# build configuration object. It is ok if it is simply "0.0".
set(KITSUNE_HIP_VERSION_STRING
  "${KITSUNE_HIP_VERSION_MAJOR}.${KITSUNE_HIP_VERSION_MINOR}")

if (KITSUNE_LAMBDA_ENABLED)
  message(FATAL_ERROR "NOT YET IMPLEMENTED: 'lambda' tapir target")
endif ()

if (KITSUNE_OMPTASK_ENABLED)
  message(FATAL_ERROR "NOT YET IMPLEMENTED: 'omptask' tapir target")
endif ()

if (KITSUNE_OPENCILK_ENABLED)
  message(STATUS "Kitsune: opencilk tapir target enabled")

  # We currently don't allow using a pre-built Cheetah. It would have to be
  # built with a "compatible" version of clang which is possible by using a
  # vanilla clang build of the same version, but it's probably not worth all the
  # error checking hassles that would be needed.
  set(KITSUNE_OPENCILK_BUILD_CONFIGURE_FLAGS ""
    CACHE STRING
    "Additional options to be passed to CMake when configuring Cheetah (the
    OpenCilk runtime). These are only used when building Cheetah as part of
    building Kitsune itself")

  set(KITSUNE_OPENCILK_BUILD_COMPILE_FLAGS ""
    CACHE STRING
    "Additional compiler flags needed to build Cheetah (the OpenCilk runtime).
    These are only used when building Cheetah as part of building Kitsune
    itself")

  set(KITSUNE_OPENCILK_BUILD_LINK_FLAGS ""
    CACHE STRING
    "Additional linker flags needed to build Cheetah (the OpenCilk runtime).
    These are only used when building Cheetah as part of building Kitsune
    itself")

  prepare_extproj_cmdline_opts("${KITSUNE_OPENCILK_BUILD_COMPILE_FLAGS}"
    KITSUNE_OPENCILK_C_FLAGS)
  prepare_extproj_cmdline_opts("${KITSUNE_OPENCILK_BUILD_COMPILE_FLAGS}"
    KITSUNE_OPENCILK_CXX_FLAGS)
  prepare_extproj_cmdline_opts("${KITSUNE_OPENCILK_BUILD_LINK_FLAGS}"
    KITSUNE_OPENCILK_LINKER_FLAGS)

  set(OPENCILK_RUNTIME_DIR
    ${KITSUNE_RESOURCE_INTDIR}/lib/${LLVM_DEFAULT_TARGET_TRIPLE})
  set(LIBOPENCILK_NAME ${SHLIB_PFX}opencilk${SHLIB_SFX})
  set(LIBOPENCILK ${OPENCILK_RUNTIME_DIR}/${LIBOPENCILK_NAME})

  # We pass some LLVM_* variables to make Cheetah behave as if it were an
  # in-tree build.
  #
  # NOTE: Cheetah expects to be installed to LLVM's install prefix, so do that
  # instead of installing it to ${KITSUNE_RESOURCE_INTDIR}.
  #
  # FIXME: Cheetah complains about not finding LLVMTestingSupport in
  # LLVM_AVAILABLE_LIBS. This variable is only set *after* LLVM has been built.
  # Since the cmake args are computed at configure time, it is empty and there
  # is nothing we can do about it. Need to find a way to set this at Kitsune
  # build time instead of configure time.
  set(KITSUNE_OPENCILK_CMAKE_ARGS
    -DCMAKE_C_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_C_FLAGS=${KITSUNE_OPENCILK_C_FLAGS}
    -DCMAKE_C_COMPILER_WORKS=1
    -DCMAKE_CXX_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_CXX_FLAGS=${KITSUNE_OPENCILK_CXX_FLAGS}
    -DCMAKE_CXX_COMPILER_WORKS=1
    -DCMAKE_EXE_LINKER_FLAGS=${KITSUNE_OPENCILK_LINKER_FLAGS}
    -DCMAKE_SHARED_LINKER_FLAGS=${KITSUNE_OPENCILK_LINKER_FLAGS}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_COLOR_DIAGNOSTICS=${CMAKE_COLOR_DIAGNOSTICS}
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
    -DCMAKE_SIZEOF_VOID_P=${CMAKE_SIZEOF_VOID_P}
    -DCHEETAH_DEFAULT_TARGET_TRIPLE=${LLVM_DEFAULT_TARGET_TRIPLE}
    -DCHEETAH_DIRECTORY_PER_TARGET=ON
    -DLLVM_AVAILABLE_LIBS=${LLVM_AVAILABLE_LIBS}
    -DLLVM_CMAKE_DIR=${CMAKE_BINARY_DIR}
    -DLLVM_COMMON_CMAKE_UTILS=${CMAKE_SOURCE_DIR}/llvm/cmake
    -DLLVM_LIBRARY_OUTPUT_INTDIR=${LLVM_LIBRARY_OUTPUT_INTDIR}
    -DLLVM_RUNTIME_OUTPUT_INTDIR=${LLVM_RUNTIME_OUTPUT_INTDIR}
    -DPACKAGE_VERSION=${PACKAGE_VERSION}
    ${KITSUNE_OPENCILK_BUILD_CONFIGURE_FLAGS})

  ExternalProject_Add(cheetah
    GIT_REPOSITORY          https://github.com/OpenCilk/cheetah.git
    GIT_TAG                 opencilk/v4.0-rc1
    PREFIX                  ${KITSUNE_EXTDEPS_BINARY_DIR}/opencilk
    INSTALL_DIR             ${CMAKE_BINARY_DIR}
    CMAKE_ARGS              ${KITSUNE_OPENCILK_CMAKE_ARGS}
    LIST_SEPARATOR          |
    DEPENDS                 clang lld llvm-config llvm-link
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD     TRUE
    USES_TERMINAL_INSTALL   TRUE
    UPDATE_DISCONNECTED     TRUE
    EXCLUDE_FROM_ALL        TRUE)

  # Cheetah will install the opencilk libraries to a directory structure similar
  # to what is shown below:
  #
  #     <PREFIX>/lib/clang/${LLVM_VERSION_MAJOR}/lib/<TRIPLE>
  #
  # Note the `clang` subdirectory in the path. We avoid using this in favor
  # of a structure similar to that shown here:
  #
  #     <PREFIX>/lib/kitsune/${LLVM_VERSION_MAJOR}/lib/<TRIPLE>
  #
  # Note the `kitsune` subdirectory in the path. Since we have no control over
  # Cheetah (unless we patch it), we install to clang's internal resource
  # directory, then copy the files over to Kitsune's internal resource
  # directory.
  add_custom_command(
    OUTPUT ${LIBOPENCILK}
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CLANG_RESOURCE_INTDIR}/lib/${LLVM_DEFAULT_TARGET_TRIPLE}
    ${OPENCILK_RUNTIME_DIR}
    DEPENDS cheetah
    COMMENT "Copying OpenCilk to ${KITSUNE_RESOURCE_INTDIR}"
    VERBATIM USES_TERMINAL)

  add_custom_target(opencilk
    DEPENDS ${LIBOPENCILK})

  # Only install the directory containing the opencilk libraries.
  # NOTE: The trailing slashes in the directory names below are required.
  install(DIRECTORY ${OPENCILK_RUNTIME_DIR}/
    DESTINATION ${KITSUNE_RESOURCE_DIR}/lib/${LLVM_DEFAULT_TARGET_TRIPLE}/
    USE_SOURCE_PERMISSIONS)

  list(APPEND KITSUNE_OPENCILK_LIBS ${LIBOPENCILK})
  list(APPEND KITRT_DEPENDS opencilk)
endif()

# KITSUNE_OPENMP_ENABLED is always true. The conditional really serves as a
# marker that the code in the body is configures the openmp tapir target.
if (KITSUNE_OPENMP_ENABLED)
  # We don't allow using a pre-built libomp because Kitsune's runtime calls
  # functions internal to libomp directly. These may change depending on the
  # version, so we only support those versions that are part of the same LLVM
  # version as Kitsune itself. In principle, we could use an existing libomp if
  # its version matched exactly, but it is probably not worth the additional
  # engineering effort to provide this flexibility.
  #
  # We build OpenMP from the same LLVM checkout from which Kitsune is being
  # built. We don't use LLVM's standard mechanisms for this because OpenMP is
  # treated as a runtime by LLVM while Kitsune is a project. This means that it
  # is difficult (and probably impossible) to ensure that libomp will be built
  # before Kitsune. libomp.so is also present in a directory that is not easily
  # accessible, so the library cannot be copied to clang's resource directory
  # in the build directory.
  message(STATUS "Kitsune: openmp tapir target enabled")

  set(KITSUNE_OPENMP_BUILD_CONFIGURE_FLAGS ""
    CACHE STRING
    "Additional options to be passed to CMake when configuring the openmp
    runtime. These are only used when building openmp as part of building
    Kitsune itself")

  set(KITSUNE_OPENMP_BUILD_COMPILE_FLAGS ""
    CACHE STRING
    "Additional compiler flags needed to build the openmp runtime. These are
    only used when building openmp as part of building Kitsune itself")

  set(KITSUNE_OPENMP_BUILD_LINK_FLAGS ""
    CACHE STRING
    "Additional linker flags needed to build the openmp runtime. These are only
    used when building openmp as part of building Kitsune itself")

  prepare_extproj_cmdline_opts("${KITSUNE_OPENMP_BUILD_COMPILE_FLAGS}"
    KITSUNE_OPENMP_C_FLAGS)
  prepare_extproj_cmdline_opts("${KITSUNE_OPENMP_BUILD_COMPILE_FLAGS}"
    KITSUNE_OPENMP_CXX_FLAGS)
  prepare_extproj_cmdline_opts("${KITSUNE_OPENMP_BUILD_LINK_FLAGS}"
    KITSUNE_OPENMP_LINKER_FLAGS)

  set(LIBOMP_NAME ${SHLIB_PFX}omp${SHLIB_SFX})
  set(LIBOMP ${KITSUNE_RESOURCE_INTDIR}/lib/${LIBOMP_NAME})

  # TODO: This configuration will only build a dynamic shared object (DSO) for
  # libomp. We would like both a DSO as well as a static archive. Currently,
  # there does not seem to be a way of building both at the same time. For now,
  # we build only the DSO, but it would be good to find a way to have both.
  #
  # This is a fairly minimal build of openmp to avoid spending time building
  # things that Kitsune does not directly use. At some point, if we feel it is
  # helpful to build, for instance, the ompt tools, the configuration options
  # here should be modified. We disable libomptarget since we will not be using
  # this build for offloading.
  set(KITSUNE_OPENMP_CMAKE_ARGS
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_C_FLAGS=${KITSUNE_OPENMP_C_FLAGS}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_CXX_FLAGS=${KITSUNE_OPENMP_CXX_FLAGS}
    -DCMAKE_EXE_LINKER_FLAGS=${KITSUNE_OPENMP_LINKER_FLAGS}
    -DCMAKE_SHARED_LINKER_FLAGS=${KITSUNE_OPENMP_LINKER_FLAGS}
    -DCMAKE_INSTALL_PREFIX=${KITSUNE_RESOURCE_INTDIR}
    -DCMAKE_INSTALL_LIBDIR=lib
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_COLOR_DIAGNOSTICS=${CMAKE_COLOR_DIAGNOSTICS}
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -DLLVM_BINARY_DIR=${CMAKE_BINARY_DIR}
    -DLLVM_ENABLE_RUNTIMES=openmp
    -DLLVM_TARGET_ARCH=${LLVM_TARGET_ARCH}
    -DLLVM_HOST_TRIPLE=${LLVM_HOST_TRIPLE}
    -DOPENMP_STANDALONE_BUILD=ON
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF
    -DOPENMP_ENABLE_LIBOMP_PROFILING=OFF
    -DOPENMP_ENABLE_OMPT_TOOLS=OFF
    -DLIBOMP_OMPD_SUPPORT=OFF
    ${KITSUNE_OPENMP_BUILD_CONFIGURE_FLAGS})

  # The openmp tests are not very cheap, so we don't run them before installing.
  # But we probably should.
  ExternalProject_Add(openmp
    PREFIX                  ${KITSUNE_EXTDEPS_BINARY_DIR}/openmp
    SOURCE_DIR              ${CMAKE_SOURCE_DIR}/../runtimes
    INSTALL_DIR             ${KITSUNE_RESOURCE_INTDIR}
    CMAKE_ARGS              ${KITSUNE_OPENMP_CMAKE_ARGS}
    LIST_SEPARATOR          |
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD     TRUE
    USES_TERMINAL_INSTALL   TRUE
    UPDATE_DISCONNECTED     TRUE
    EXCLUDE_FROM_ALL        TRUE
    INSTALL_BYPRODUCTS      ${LIBOMP})

  # When installing Kitsune, we only install libomp because we don't need
  # anything else.
  # NOTE: The trailing slashes in the directory names below are required.
  install(DIRECTORY ${KITSUNE_RESOURCE_INTDIR}/lib/
    DESTINATION ${KITSUNE_RESOURCE_DIR}/lib/
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING PATTERN "${SHLIB_PFX}omp*")

  ExternalProject_Get_Property(openmp BINARY_DIR)
  list(APPEND KITSUNE_OPENMP_INCDIRS
    ${LLVM_MONOREPO_SOURCE_DIR}/openmp/runtime/src
    ${BINARY_DIR}/openmp/runtime/src)
  list(APPEND KITSUNE_OPENMP_LIBS ${LIBOMP})
  list(APPEND KITRT_DEPENDS openmp)
endif ()

if (KITSUNE_QTHREADS_ENABLED)
  # We currently don't allow using a pre-built qthreads. It should be ok to do
  # this since it is not likely that the interface, or behavior, will change.
  # Since this library is unlikely to be present on the average system on which
  # we expect Kitsune to be used, we are not particularly keen on supporting
  # this anytime soon.
  message(STATUS "Kitsune: qthreads tapir target enabled")

  set(KITSUNE_QTHREADS_BUILD_CONFIGURE_FLAGS ""
    CACHE STRING
    "Additional options to be passed to CMake when configuring Qthreads. These
    are only used when building Qthreads as part of building Kitsune itself")

  set(KITSUNE_QTHREADS_BUILD_COMPILE_FLAGS ""
    CACHE STRING
    "Additional compiler flags needed to build Qthreads. These are only used
    when building Qthreads as part of building Kitsune itself")

  set(KITSUNE_QTHREADS_BUILD_LINK_FLAGS ""
    CACHE STRING
    "Additional linker flags needed to build Qthreads. These are only used when
    building Qthreads as part of building Kitsune itself")

  prepare_extproj_cmdline_opts("${KITSUNE_QTHREADS_BUILD_COMPILE_FLAGS}"
    KITSUNE_QTHREADS_C_FLAGS)
  prepare_extproj_cmdline_opts("${KITSUNE_QTHREADS_BUILD_COMPILE_FLAGS}"
    KITSUNE_QTHREADS_CXX_FLAGS)
  prepare_extproj_cmdline_opts("${KITSUNE_QTHREADS_BUILD_LINK_FLAGS}"
    KITSUNE_QTHREADS_LINKER_FLAGS)

  set(LIBQTHREAD ${KITSUNE_RESOURCE_INTDIR}/lib/${SHLIB_PFX}qthread${SHLIB_SFX})

  set(KITSUNE_QTHREADS_CMAKE_ARGS
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_C_FLAGS=${KITSUNE_QTHREADS_C_FLAGS}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_CXX_FLAGS=${KITSUNE_QTHREADS_CXX_FLAGS}
    -DCMAKE_EXE_LINKER_FLAGS=${KITSUNE_QTHREADS_LINKER_FLAGS}
    -DCMAKE_SHARED_LINKER_FLAGS=${KITSUNE_QTHREADS_LINKER_FLAGS}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_COLOR_DIAGNOSTICS=${CMAKE_COLOR_DIAGNOSTICS}
    -DCMAKE_INSTALL_PREFIX=${KITSUNE_RESOURCE_INTDIR}
    -DCMAKE_INSTALL_LIBDIR=lib
    -DCMAKE_INSTALL_INCLUDEDIR=include/qthread
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -DQTHREADS_TOPOLOGY=hwloc
    -DQTHREADS_SCHEDULER=distrib
    ${KITSUNE_QTHREADS_BUILD_CONFIGURE_FLAGS})

  ExternalProject_Add(qthreads
    GIT_REPOSITORY          https://github.com/sandialabs/qthreads.git
    GIT_TAG                 1.23
    PREFIX                  ${KITSUNE_EXTDEPS_BINARY_DIR}/qthreads
    INSTALL_DIR             ${KITSUNE_RESOURCE_INTDIR}
    CMAKE_ARGS              ${KITSUNE_QTHREADS_CMAKE_ARGS}
    LIST_SEPARATOR          |
    TEST_BEFORE_INSTALL     TRUE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD     TRUE
    USES_TERMINAL_INSTALL   TRUE
    UPDATE_DISCONNECTED     TRUE
    EXCLUDE_FROM_ALL        TRUE
    INSTALL_BYPRODUCTS      ${LIBQTHREAD})

  # When installing Kitsune, only install libqthread because we don't need
  # anything else.
  # NOTE: The trailing slashes in the directory names below are required.
  install(DIRECTORY ${KITSUNE_RESOURCE_INTDIR}/lib/
    DESTINATION ${KITSUNE_RESOURCE_DIR}/lib/
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING PATTERN "${SHLIB_PFX}qthread*"
    PATTERN "cmake" EXCLUDE)

  list(APPEND KITSUNE_QTHREADS_INCDIRS ${KITSUNE_RESOURCE_INTDIR}/include)
  list(APPEND KITSUNE_QTHREADS_LIBS ${LIBQTHREAD})
  list(APPEND KITRT_DEPENDS qthreads)
endif()

if (KITSUNE_REALM_ENABLED)
  message(FATAL_ERROR "NOT YET IMPLEMENTED: 'realm' tapir target")
endif()

# The name of Kitsune's main runtime library. This is created here because it is
# also used by the tests to ensure that check-kitsune does not run before the
# runtime has been built.
set(KITRT_DSO "kitrt")
set(KITRT_STATIC "kitrt_static")

# Notes for the subdirectories below
#
#   - The order in which the subdirectories are entered is important. Those
#     that are entered later may depend on one or more that are entered earlier.
#
#   - We do not enter the lib subdirectory here. The contents of lib/ are
#     treated as core LLVM libraries which become part of libLLVM.so. Therefore,
#     the contents are handled in llvm/lib/CMakeLists.txt.
#
# TODO: Take a closer look at cmake/modules and see if anything there is
# actually relevant. It is not clear how useful that is.
add_subdirectory(cmake/modules)
add_subdirectory(include/kitsune)
add_subdirectory(tools)
add_subdirectory(runtime)

if (KITSUNE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif ()

if (KITSUNE_ENABLE_SPHINX OR KITSUNE_ENABLE_DOXYGEN)
  add_subdirectory(docs)
endif ()

if(LLVM_INCLUDE_TESTS AND KITSUNE_INCLUDE_TESTS)
  add_subdirectory(runtime/unittests)
  add_subdirectory(runtime/test)
  add_subdirectory(unittests)
  add_subdirectory(test)
endif()

if (KITSUNE_ENABLE_EXPERIMENTS)
  add_subdirectory(experiments)
endif ()

# Create the rules to create a "base" Kitsune executable. This is simply a copy
# of the base clang (or flang) executable. We have to create a copy because a
# symlink will not work as we want. Very early in the process, the symlinks are
# stripped away and the underlying executable is exposed to the driver. We need
# this to *not* be named clang in order to determine whether we are compiling
# with a kitsune frontend. To be consistent with clang, we do not create a
# separate frontend for C++.
#
# CLONE is the name of the executable that will be created. This will be an
# exact copy of the executable corresponding to the CMake target BASE. BASE will
# be either clang, or flang.
function(setup_frontend_main base clone)
  set(MAIN "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${clone}")

  add_custom_command(
    OUTPUT ${MAIN}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${base}> ${MAIN}
    DEPENDS ${base}
    COMMENT "Creating ${clone}"
    VERBATIM USES_TERMINAL)

  install(PROGRAMS ${MAIN}
    DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction ()

# Create a symlink to the main executable that was created by calling
# `setup_frontend_main`. The symlink will be named NAME and live in the same
# directory as the main executable, TARGET. CMAKE_TARGET is the name of the
# custom target that will be created to drive the creation of the symlink. Other
# entities will depend on CMAKE_TARGET, which will be added to ALL.
function(setup_frontend_symlink name target cmake_target)
  set(MAIN ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target})
  set(SYMLINK ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name})

  # We want the symlink to target the main executable that is known to live
  # next to it. Therefore, we only use the name of the target in create_symlink
  # command, not its full path. If we use the latter, the symlink will continue
  # to point to the executable in the build directory even after installation.
  add_custom_command(
    OUTPUT ${SYMLINK}
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${target} ${SYMLINK}
    DEPENDS ${MAIN}
    COMMENT "Creating symlink ${name} to ${target}"
    VERBATIM USES_TERMINAL)

  install(PROGRAMS ${SYMLINK}
    DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_custom_target(${cmake_target} ALL DEPENDS ${SYMLINK})
endfunction()

# Create an executable named kitcc-${LLVM_VERSION_MAJOR} which is a copy of
# clang-${LLVM_VERSION_MAJOR}. Then create symlinks to this executable named
# ${KITSUNE_C_FRONTEND}, and ${KITSUNE_CXX_FRONTEND}.
set(KITSUNE_CXX_COMPILER kitcc-${LLVM_VERSION_MAJOR})
setup_frontend_main(clang ${KITSUNE_CXX_COMPILER})
setup_frontend_symlink(${KITSUNE_C_FRONTEND} ${KITSUNE_CXX_COMPILER} kitcc)
setup_frontend_symlink(${KITSUNE_CXX_FRONTEND} ${KITSUNE_CXX_COMPILER} kit++)

if (KITSUNE_Fortran_ENABLED)
  # Create an executable named kitfc-${LLVM_VERSION_MAJOR} which is a copy of
  # flang-${LLVM_VERSION_MAJOR}. Then create a symlink to this executable named
  # ${KITSUNE_Fortran_FRONTEND}.
  set(KITSUNE_Fortran_COMPILER kitfc-${LLVM_VERSION_MAJOR})
  setup_frontend_main(flang ${KITSUNE_Fortran_COMPILER})
  setup_frontend_symlink(${KITSUNE_Fortran_FRONTEND} ${KITSUNE_Fortran_COMPILER} kitfc)
endif()
