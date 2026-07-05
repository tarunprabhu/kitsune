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
# When building Kitsune's dependencies such as Kokkos and Cheetah - and also
# Kitsune's own runtime, kitrt - we suppress CMake's checks that the compiler
# works by setting -DCMAKE_CXX_COMPILER_WORKS=1. These are built with the clang
# that has just been built. They are not, and must not be, built with the
# compiler being used to build Kitsune itself.
#
# On systems where clang requires --sysroot to function correctly, the compiler
# checks will fail since they do not set --sysroot. For now, a workaround is to
# suppress those checks. It is reasonable to expect that the clang that was just
# build works - if it doesn't, then we obviously have bigger problems (if it
# doesn't work, the cause should nearly always be a bug introduced by a Kitsune
# developer who will, naturally, fix the issue immediately). We ensure that
# --sysroot=${KITSUNE_SYSROOT} is added to the build commands for the
# dependencies and kitrt, so, barring other issues, the build will succeed.
#
# TODO: A better fix might be to generate a configuration file in this case just
# for the purpose of building the tool/runtime with clang with --sysroot set.
# But this might result in other issues related to the tests. It would be
# worth exploring this option though.
#
# The dependencies that are built together with Kitsune i.e. Cheetah, Kokkos
# etc. (but not cuda and hip) are always installed to
# ${CLANG_RESOURCE_INSTALL_DIR}. This effectively makes these dependencies
# "internal" to Kitsune. This is intentional. We currently do not support
# using an external build of these dependencies since the tapir targets and
# Kitsune's own runtime expects specific interfaces and we do not intend to
# support ranges of libraries. If we were to install these to
# ${CMAKE_INSTALL_PREFIX}, i.e. the top-level install directory for Kitsune
# itself, we may conflict with versions of these libraries that may be installed
# to the same prefix as Kitsune.
#
# ------------------------------------------------------------------------------

# Lookup the unique directories containing the libraries in LIBS and add them
# to the list OUT.
macro(populate_unique_parents LIBS OUT)
  foreach (LIB IN LISTS ${LIBS})
    get_filename_component(DIR ${LIB} DIRECTORY)
    if (NOT DIR IN_LIST ${OUT})
      list(APPEND ${OUT} ${DIR})
    endif ()
  endforeach ()
endmacro ()

# Prepare the variable OUT for use in an ExternalProject command. INP is a list
# of compiler/linker options. All elements of INP will be copied to OUT. In
# addition to this, the --sysroot= option will be added to OUT if
# KITSUNE_SYSROOT is not empty. OUT will use '|' as the list separator instead
# of the usual ';'. Because ExternalProject issues commands on a terminal, the
# use of ';' is unreliable since the terminal may interpret it as an
# end-of-command marker. '|' was deemed to be safe since it is unlikely to
# appear in the library or path names.
macro(prepare_extproj_cmdline_opts INP OUT)
  set(${OUT} "${INP}")
  if (KITSUNE_SYSROOT)
    string(APPEND ${OUT} " --sysroot=${KITSUNE_SYSROOT}")
  endif ()
  string(REPLACE ";" "|" ${OUT} "${${OUT}}")
endmacro ()

# FIXME: Should this be named KITRT_* since we don't have a separate kitcuda
# library any longer?
option(KITCUDA_ENABLE_NVTX
  "Enable NVTX profiling within the runtime"
  OFF)

# Convenience variables that are shorter.
set(SHLIB_PFX ${CMAKE_SHARED_LIBRARY_PREFIX})
set(SHLIB_SFX ${CMAKE_SHARED_LIBRARY_SUFFIX})
set(ARCHIVE_PFX ${CMAKE_STATIC_LIBRARY_PREFIX})
set(ARCHIVE_SFX ${CMAKE_STATIC_LIBRARY_SUFFIX})

# The name of Kitsune's main runtime library. This is created here because it is
# also used by the tests to ensure that check-kitsune does not run before the
# runtime has been built.
set(KITRT kitrt)
set(KITRT_STATIC "${KITRT}_static")
set(LIBNAME_KITRT ${SHLIB_PFX}${KITRT}${SHLIB_SFX})
set(LIBNAME_KITRT_STATIC ${ARCHIVE_PFX}${KITRT_STATIC}${ARCHIVE_SFX})

set(KITSUNE_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(KITSUNE_INCLUDE_DIR ${KITSUNE_SOURCE_DIR}/include)
set(KITSUNE_BINARY_DIR  ${PROJECT_BINARY_DIR})
set(KITSUNE_TARGETS_BINARY_DIR ${KITSUNE_BINARY_DIR}/targets)
set(KITSUNE_OPTDEPS_BINARY_DIR ${KITSUNE_BINARY_DIR}/optdeps)

# Dependencies for Kitsune's runtime. lld is only needed if
# -DLLVM_USE_LINKER=lld is provided at configure-time, but adding it as a
# dependency unconditionally feels "safer".
set(KITRT_DEPENDS clang lld llvm-ar llvm-config llvm-link LLVM)

# Just in case, check for utilities that are needed to build some of the
# dependencies. These are not needed for everything, but they are basic enough
# that we might as well check for them unconditionally and "globally".
find_program(GIT git REQUIRED)
find_program(GREP grep REQUIRED)
find_program(PATCH patch REQUIRED)
find_program(WC wc REQUIRED)

# The clang resource directories in the build directory.
get_clang_resource_dir(CLANG_RESOURCE_INTDIR
  PREFIX ${CMAKE_BINARY_DIR})
get_clang_resource_dir(CLANG_RESOURCE_INT_LIBDIR
  PREFIX ${CMAKE_BINARY_DIR}
  SUBDIR ${CMAKE_INSTALL_LIBDIR})

# The clang resource directories after installation.
get_clang_resource_dir(CLANG_RESOURCE_INSTALL_DIR
  PREFIX ${CMAKE_INSTALL_PREFIX})
get_clang_resource_dir(CLANG_RESOURCE_INSTALL_LIBDIR
  PREFIX ${CMAKE_INSTALL_PREFIX}
  SUBDIR ${CMAKE_INSTALL_LIBDIR})

find_library(LIB_DL dl REQUIRED)
list(APPEND KITSUNE_COMMON_LIB_NAMES dl)
list(APPEND KITSUNE_COMMON_LIBS ${LIB_DL})

find_library(LIB_M m REQUIRED)
list(APPEND KITSUNE_COMMON_LIB_NAMES m)
list(APPEND KITSUNE_COMMON_LIBS ${LIB_M})

# The pthread functions are now part of libc. libpthread was removed from glibc
# 2.34. There is no need to explicitly link this in unless we have older
# versions of libc around. We should consider removing this from here at some
# point when we are certain we don't need this. That might be a while since HPC
# machines are notorious for their antique software stacks.
#
# https://developers.redhat.com/articles/2021/12/17/why-glibc-234-removed-libpthread
#
find_library(LIB_PTHREAD pthread REQUIRED)
list(APPEND KITSUNE_COMMON_LIB_NAMES pthread)
list(APPEND KITSUNE_COMMON_LIBS ${LIB_PTHREAD})

# The librt functions are part of glibc as of v2.34. There is no need to link
# this in explicitly unless we have older versions of libc around. We do have to
# suppport HPC clusters with old libc installations, so this might have to be
# around for some time yet.
find_library(LIB_RT rt REQUIRED)
list(APPEND KITSUNE_COMMON_LIB_NAMES rt)
list(APPEND KITSUNE_COMMON_LIBS ${LIB_RT})

# On FreeBSD, the backtrace-related functions are defined in libexecinfo. On
# other systems, they are part of libc.
if (CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
  find_library(LIB_EXECINFO execinfo REQUIRED)
  list(APPEND KITSUNE_COMMON_LIB_NAMES execinfo)
  list(APPEND KITSUNE_COMMON_LIBS ${LIB_EXECINFO})
endif()

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

  set(KITSUNE_PAPI_SOURCE_DIR ${KITSUNE_OPTDEPS_BINARY_DIR}/papi)
  set(KITSUNE_PAPI_INSTALL_DIR ${KITSUNE_OPTDEPS_BINARY_DIR}/papi/install)
  set(KITSUNE_PAPI_CONFIGURE_FLAGS
    --disable-fortran
    --with-shared-lib=no
    --with-static-lib=yes
    --prefix=${KITSUNE_PAPI_INSTALL_DIR})
  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND KITSUNE_PAPI_CONFIGURE_FLAGS --with-perf-events)
  endif ()

  # If we don't specify the compiler explicitly, it will pick whatever is on
  # ${PATH}. This may not be desirable, for instance on HPC systems, where we
  # might just end up with some antique compiler.
  set(CC ${CMAKE_C_COMPILER})

  # PAPI-7.2.0 fails to build if -std=c23 is used. This is the default since
  # GCC 15, so set the standard explicitly. It is not clear if PAPI builds with
  # -fPIC automatically, so add it explicitly.
  set(CFLAGS "-std=gnu17 -fPIC")

  # Notes about the commands below:
  #
  #   - The configure script must be run from the src/ directory within PAPI.
  #     There might be a way to get `make` to do the right thing if it is run
  #     from elsewhere, but if there is, I haven't found it.
  #
  #   - BUILD_COMMAND assumes that `make` is GNU make since it uses `-j`
  #     without a value, as is required by, for instance, BSD make.
  #
  # The GIT_TAG must be updated when we wish to use a newer version of PAPI.
  ExternalProject_Add(papi
    GIT_REPOSITORY          https://github.com/icl-utk-edu/papi
    GIT_TAG                 papi-7-2-0-t
    BUILD_IN_SOURCE         TRUE
    SOURCE_DIR              ${KITSUNE_PAPI_SOURCE_DIR}
    CONFIGURE_COMMAND       cd ${KITSUNE_PAPI_SOURCE_DIR}/src &&
                            ${CMAKE_COMMAND} -E env
                            CC=${CC} "CFLAGS=${CFLAGS}" CONFIG_SHELL=${BASH}
                            --
                            ./configure ${KITSUNE_PAPI_CONFIGURE_FLAGS}
    BUILD_COMMAND           ${MAKE} -C ${KITSUNE_PAPI_SOURCE_DIR}/src -j
    INSTALL_COMMAND         ${MAKE} -C ${KITSUNE_PAPI_SOURCE_DIR}/src install
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD     TRUE
    USES_TERMINAL_INSTALL   TRUE
    UPDATE_DISCONNECTED     TRUE
    EXCLUDE_FROM_ALL        TRUE)

  set(LIBNAME_PAPI ${ARCHIVE_PFX}papi${ARCHIVE_SFX})

  # We don't need to copy PAPI to a staging area because we only build a static
  # library that is linked directly into Kitsune's runtime when it is built.
  set(KITSUNE_PAPI_INCDIRS ${KITSUNE_PAPI_INSTALL_DIR}/include)
  set(KITSUNE_PAPI_LIBS ${KITSUNE_PAPI_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/${LIBNAME_PAPI})

  list(APPEND KITRT_DEPENDS papi)
endif ()

if (KITSUNE_KOKKOS_ENABLED)
  # We currently don't support using a pre-built Kokkos - even if it has been
  # patched the way we need it to be. This could possibly be moved to
  # kitsune/runtime, just so everything that needs to be fetched and built is
  # there, even if this is not technically a runtime.

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

  set(KITSUNE_KOKKOS_SOURCE_DIR  ${KITSUNE_TARGETS_BINARY_DIR}/kokkos/kokkos)
  set(KITSUNE_KOKKOS_BUILD_DIR ${KITSUNE_TARGETS_BINARY_DIR}/kokkos/build)
  set(KITSUNE_KOKKOS_INSTALL_DIR ${KITSUNE_TARGETS_BINARY_DIR}/kokkos/install)

  # Here, we set Kokkos_ENABLE_SERIAL to ON, and don't build any parallel
  # "backends". This is intentional. We build Kokkos ourselves because we have
  # to patch it to remove exceptions. It isn't used for anything except for
  # testing Kitsune's Kokkos-mode - and even that is just a demonstration.
  set(KITSUNE_KOKKOS_BUILD_CMAKE_FLAGS
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_CXX_FLAGS=${KITSUNE_KOKKOS_CXX_FLAGS}
    -DCMAKE_EXE_LINKER_FLAGS=${KITSUNE_KOKKOS_LINKER_FLAGS}
    -DCMAKE_SHARED_LINKER_FLAGS=${KITSUNE_KOKKOS_LINKER_FLAGS}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_COLOR_DIAGNOSTICS=${CMAKE_COLOR_DIAGNOSTICS}
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -DCMAKE_INSTALL_PREFIX=${KITSUNE_KOKKOS_INSTALL_DIR}
    -DKokkos_ENABLE_SERIAL=ON
    -DBUILD_TESTING=OFF
    -DBUILD_SHARED_LIBS=ON
    ${KITSUNE_KOKKOS_BUILD_CONFIGURE_FLAGS})

  ExternalProject_Add(kokkos
    GIT_REPOSITORY          https://github.com/kokkos/kokkos.git
    GIT_TAG                 4.7.00
    SOURCE_DIR              ${KITSUNE_KOKKOS_SOURCE_DIR}
    BINARY_DIR              ${KITSUNE_KOKKOS_BUILD_DIR}
    INSTALL_DIR             ${KITSUNE_KOKKOS_INSTALL_DIR}
    CMAKE_ARGS              ${KITSUNE_KOKKOS_BUILD_CMAKE_FLAGS}
    LIST_SEPARATOR          |
    USES_TERMINAL_PATCH     TRUE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD     TRUE
    USES_TERMINAL_INSTALL   TRUE
    UPDATE_DISCONNECTED     TRUE
    EXCLUDE_FROM_ALL        TRUE)

  # These are the only headers and libraries used in Kitsune's Kokkos-mode demos
  # and tests.
  set(LIBNAME_KOKKOSCORE ${SHLIB_PFX}kokkoscore${SHLIB_SFX})
  set(KITSUNE_KOKKOS_LIB_KOKKOSCORE
    ${CLANG_RESOURCE_INTDIR}/kokkos/${CMAKE_INSTALL_LIBDIR}/${LIBNAME_KOKKOSCORE})
  set(KITSUNE_KOKKOS_CORE_H
    ${CLANG_RESOURCE_INTDIR}/kokkos/include/Kokkos_Core.hpp)

  # Copy the Kokkos files into a "staging" area. This is needed to run the
  # kitsune tests and also to allow kitsune to be run from the build directory.
  add_custom_command(
    OUTPUT ${KITSUNE_KOKKOS_CORE_H} ${KITSUNE_KOKKOS_LIB_CORE}
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${KITSUNE_KOKKOS_INSTALL_DIR}/include
    ${CLANG_RESOURCE_INTDIR}/kokkos/include
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${KITSUNE_KOKKOS_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}
    ${CLANG_RESOURCE_INTDIR}/kokkos/${CMAKE_INSTALL_LIBDIR}
    DEPENDS kokkos
    COMMENT "Copying Kokkos to the top-level build directory"
    VERBATIM USES_TERMINAL)

  # This target drives the command to "install" Cheetah to the build directory.
  add_custom_target(kokkos-copy-to-build-dir
    DEPENDS
    ${KITSUNE_KOKKOS_CORE_H}
    ${KITSUNE_KOKKOS_LIB_CORE})

  # The runtime does not actually depend on this, but we need some way to run
  # the target since it is excluded from ALL.
  list(APPEND KITRT_DEPENDS kokkos-copy-to-build-dir)

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
  install(DIRECTORY ${KITSUNE_KOKKOS_INSTALL_DIR}/
    DESTINATION ${CLANG_RESOURCE_INSTALL_DIR}/kokkos
    USE_SOURCE_PERMISSIONS
    PATTERN "cmake" EXCLUDE
    PATTERN "bin" EXCLUDE)
endif()

set(KITSUNE_CUDA_VERSION_MAJOR 0)
set(KITSUNE_CUDA_VERSION_MINOR 0)
set(KITSUNE_CUDA_VERSION_PATCH 0)
if (KITSUNE_CUDA_ENABLED)
  message(STATUS "Kitsune: cuda tapir target enabled")

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
  set(KITSUNE_CUDA_VERSION_MAX 13.3)

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

  # This should almost always be the libcuda that is installed on the system,
  # so look for it in the "usual" places.
  find_library(KITSUNE_CUDA_LIB_CUDA cuda REQUIRED)
  list(APPEND KITSUNE_CUDA_LIB_NAMES cuda)
  list(APPEND KITSUNE_CUDA_LIBS ${KITSUNE_CUDA_LIB_CUDA})

  find_library(KITSUNE_CUDA_LIB_CUDART_STATIC cudart_static REQUIRED
    PATHS ${CUDAToolkit_LIBRARY_DIR}
    NO_DEFAULT_PATH)
  list(APPEND KITSUNE_CUDA_LIB_NAMES cudart_static)
  list(APPEND KITSUNE_CUDA_LIBS ${KITSUNE_CUDA_LIB_CUDART_STATIC})

  find_program(KITSUNE_CUDA_PTXAS ptxas REQUIRED
    PATHS ${CUDAToolkit_BIN_DIR}
    NO_DEFAULT_PATH)

  find_program(KITSUNE_CUDA_FATBINARY fatbinary REQUIRED
    PATHS ${CUDAToolkit_BIN_DIR}
    NO_DEFAULT_PATH)

  if (KITCUDA_ENABLE_NVTX)
    find_library(KITSUNE_CUDA_LIB_NVTOOLS_EXT nvToolsExt REQUIRED
      PATHS ${KITSUNE_CUDA_LIBDIRS}
      NO_DEFAULT_PATH)
    list(APPEND KITSUNE_CUDA_LIB_NAMES nvToolsExt)
    list(APPEND KITSUNE_CUDA_LIBS ${KITSUNE_CUDA_LIB_NVTOOLS_EXT})

    set(KITSUNE_CUDA_DEFNS KITCUDA_ENABLE_NVTX)
  endif()

  populate_unique_parents(KITSUNE_CUDA_LIBS KITSUNE_CUDA_LIBDIRS)
  string(REPLACE ";" "|" KITSUNE_CUDA_DEFNS "${KITSUNE_CUDA_DEFNS}")
  string(REPLACE ";" "|" KITSUNE_CUDA_LIB_NAMES "${KITSUNE_CUDA_LIB_NAMES}")
  string(REPLACE ";" "|" KITSUNE_CUDA_LIBS "${KITSUNE_CUDA_LIBS}")
  string(REPLACE ";" "|" KITSUNE_CUDA_LIBDIRS "${KITSUNE_CUDA_LIBDIRS}")
  string(REPLACE ";" "|" KITSUNE_CUDA_INCDIRS "${CUDAToolkit_INCLUDE_DIRS}")
  string(REPLACE ";" "|" KITSUNE_CUDA_BIN_DIRS "${CUDAToolkit_BIN_DIR}")

  set(KITSUNE_CUDA_ARCH_DEFAULT "sm_80"
    CACHE STRING
    "The default NVIDIA GPU architecture used by the cuda tapir target when a GPU is not detected on the system")
endif()
set(KITSUNE_CUDA_VERSION_STRING "${KITSUNE_CUDA_VERSION_MAJOR}.${KITSUNE_CUDA_VERSION_MINOR}.${KITSUNE_CUDA_VERSION_PATCH}")

set(KITSUNE_HIP_VERSION_MAJOR 0)
set(KITSUNE_HIP_VERSION_MINOR 0)
set(KITSUNE_HIP_VERSION_PATCH 0)
if (KITSUNE_HIP_ENABLED)
  message(STATUS "Kitsune: hip tapir target enabled.")

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

  find_library(KITSUNE_HIP_LIB_AMDHIP64 amdhip64 REQUIRED
    PATHS ${HIP_LIB_INSTALL_DIR}
    NO_DEFAULT_PATH)
  list(APPEND KITSUNE_HIP_LIB_NAMES amdhip64)
  list(APPEND KITSUNE_HIP_LIBS ${KITSUNE_HIP_LIB_AMDHIP64})

  populate_unique_parents(KITSUNE_HIP_LIBS KITSUNE_HIP_LIBDIRS)
  string(REPLACE ";" "|" KITSUNE_HIP_LIB_NAMES "${KITSUNE_HIP_LIB_NAMES}")
  string(REPLACE ";" "|" KITSUNE_HIP_LIBS "${KITSUNE_HIP_LIBS}")
  string(REPLACE ";" "|" KITSUNE_HIP_LIBDIRS "${KITSUNE_HIP_LIBDIRS}")
  string(REPLACE ";" "|" KITSUNE_HIP_INCDIRS "${HIP_INCLUDE_DIR}")

  set(KITSUNE_HIP_ARCH_DEFAULT "gfx90a"
    CACHE STRING
    "The default AMD GPU architecture used by the hip tapir target when a GPU is not detected on the system")

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
set(KITSUNE_HIP_VERSION_STRING "${KITSUNE_HIP_VERSION_MAJOR}.${KITSUNE_HIP_VERSION_MINOR}.${KITSUNE_HIP_VERSION_PATCH}")

if (KITSUNE_LAMBDA_ENABLED)
  message(FATAL_ERROR "NOT YET IMPLEMENTED: 'lambda' tapir target")
  message(STATUS "Kitsune: lambda tapir target enabled")
endif ()

if (KITSUNE_OMPTASK_ENABLED)
  message(FATAL_ERROR "NOT YET IMPLEMENTED: 'omptask' tapir target")
  message(STATUS "Kitsune: omptask target enabled")
endif ()

if (KITSUNE_OPENCILK_ENABLED)
  message(STATUS "Kitsune: opencilk target enabled")

  # We currently don't allow using a pre-built Cheetah. It would have to be
  # built with a "compatible" version of clang which is possible by using a
  # vanilla clang build of the same version, but it's probably not worth all the
  # error checking hassles that would be needed to detect if the versions were
  # mismatched.
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

  # We pass some LLVM_* variables to make Cheetah behave as if it were an
  # in-tree build.
  #
  # FIXME: Cheetah complains about not finding LLVMTestingSupport in
  # LLVM_AVAILABLE_LIBS. This variable is only set *after* LLVM has been built.
  # Since the cmake args are computed at configure time, it is empty and there
  # is nothing we can do about it. Need to find a way to set this at Kitsune
  # build time instead of configure time.
  set(KITSUNE_OPENCILK_SOURCE_DIR ${KITSUNE_TARGETS_BINARY_DIR}/opencilk/cheetah)
  set(KITSUNE_OPENCILK_BINARY_DIR ${KITSUNE_TARGETS_BINARY_DIR}/opencilk/build)
  set(KITSUNE_OPENCILK_INSTALL_DIR ${KITSUNE_TARGETS_BINARY_DIR}/opencilk/install)
  set(KITSUNE_OPENCILK_BUILD_CMAKE_FLAGS
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
    -DCMAKE_INSTALL_PREFIX=${KITSUNE_OPENCILK_INSTALL_DIR}
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
    DEPENDS                 clang lld llvm-config llvm-link
    GIT_REPOSITORY          https://github.com/OpenCilk/cheetah.git
    GIT_TAG                 opencilk/v4.0-rc1
    SOURCE_DIR              ${KITSUNE_OPENCILK_SOURCE_DIR}
    BINARY_DIR              ${KITSUNE_OPENCILK_BINARY_DIR}
    INSTALL_DIR             ${KITSUNE_OPENCILK_INSTALL_DIR}
    CMAKE_ARGS              ${KITSUNE_OPENCILK_BUILD_CMAKE_FLAGS}
    LIST_SEPARATOR          |
    USES_TERMINAL_PATCH     TRUE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD     TRUE
    USES_TERMINAL_INSTALL   TRUE
    UPDATE_DISCONNECTED     TRUE
    EXCLUDE_FROM_ALL        TRUE)

  set(KITSUNE_OPENCILK_SUBDIR
    ${CLANG_RESOURCE_INT_LIBDIR}/${LLVM_DEFAULT_TARGET_TRIPLE})

  set(BCNAME_OPENCILK ${SHLIB_PFX}opencilk-abi.bc)
  set(KITSUNE_OPENCILK_BC ${KITSUNE_OPENCILK_SUBDIR}/${BCNAME_OPENCILK})

  set(LIBNAME_OPENCILK ${SHLIB_PFX}opencilk${SHLIB_SFX})
  set(KITSUNE_OPENCILK_LIB_OPENCILK
    ${KITSUNE_OPENCILK_SUBDIR}/${LIBNAME_OPENCILK})

  set(LIBNAME_PERSONALITY_C ${SHLIB_PFX}opencilk-personality-c${SHLIB_SFX})
  set(KITSUNE_OPENCILK_LIB_PERSONALITY_C
    ${KITSUNE_OPENCILK_SUBDIR}/${LIBNAME_PERSONALITY_C})

  set(LIBNAME_PERSONALITY_CXX ${SHLIB_PFX}opencilk-personality-cpp${SHLIB_SFX})
  set(KITSUNE_OPENCILK_LIB_PERSONALITY_CXX
    ${KITSUNE_OPENCILK_SUBDIR}/${LIBNAME_PERSONALITY_CXX})

  set(KITSUNE_OPENCILK_LIB_PEDIGREES
    ${KITSUNE_OPENCILK_SUBDIR}/libopencilk-pedigrees.so)

  # "Install" the required cheetah files into the top-level build directory.
  # This allows kitsune to be run from the build directory. The destination
  # directory is set to ${CMAKE_BINARY_DIR} in the command below because
  # Cheetah's build system assumes that it will be installed to the top-level
  # directory - this way, Cheetah's libraries and headers will end up in clang's
  # resource directory.
  add_custom_command(
    OUTPUT
    ${KITSUNE_OPENCILK_BC}
    ${KITSUNE_OPENCILK_LIB_OPENCILK}
    ${KITSUNE_OPENCILK_LIB_PERSONALITY_C}
    ${KITSUNE_OPENCILK_LIB_PERSONALITY_CXX}
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${KITSUNE_OPENCILK_INSTALL_DIR}
    ${CMAKE_BINARY_DIR}
    DEPENDS cheetah
    COMMENT "Copying OpenCilk to the top-level build directory"
    VERBATIM USES_TERMINAL)

  # This target drives the command to "install" Cheetah to the build directory.
  add_custom_target(opencilk-copy-to-build-dir
    DEPENDS
    ${KITSUNE_OPENCILK_BC}
    ${KITSUNE_OPENCILK_LIB_OPENCILK}
    ${KITSUNE_OPENCILK_LIB_PERSONALITY_C}
    ${KITSUNE_OPENCILK_LIB_PERSONALITY_CXX})

  # Add this to the dependencies for Kitsune. This will ensure that OpenCilk
  # gets built. We have excluded it from ALL, so it won't be built
  # automatically.
  list(APPEND KITRT_DEPENDS opencilk-copy-to-build-dir)

  # When installing Kitsune, make sure to install Cheetah as well. Cheetah is to
  # be installed to ${CLANG_RESOURCE_INSTALL_DIR}. Cheetah's build system
  # ensures that the directory structure in ${KITSUNE_OPENCILK_INSTALL_DIR} is
  # such that simply copying the contents of ${KITSUNE_OPENCILK_INSTALL_DIR} to
  # ${CMAKE_INSTALL_PREFIX} will result in all the files being in the correct
  # locations.
  #
  # NOTE: The trailing slash after the directory to copy is required.
  install(DIRECTORY ${KITSUNE_OPENCILK_INSTALL_DIR}/
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    USE_SOURCE_PERMISSIONS
    PATTERN "cmake" EXCLUDE
    PATTERN "include" EXCLUDE)

  set(KITSUNE_OPENCILK_LIBS
    ${KITSUNE_OPENCILK_LIB_OPENCILK})
  string(REPLACE ";" "|" KITSUNE_OPENCILK_LIBS "${KITSUNE_OPENCILK_LIBS}")
endif()

# TODO: KITSUNE_OPENMP_ENABLED is always true. At some point, we might move
# the setup of the dependencies to their own files since this particular file
# is becoming fairly unwieldy. Until then, the conditional really serves as a
# marker that the code in the body is part of the configuration of the openmp
# tapir target.
if (KITSUNE_OPENMP_ENABLED)
  # We build OpenMP from the same LLVM checkout from which Kitsune is being
  # built. We don't use LLVM's standard mechanisms for this because OpenMP is
  # treated as a runtime by LLVM while Kitsune is a project. This means that it
  # is difficult (and probably impossible) to ensure that libomp will be built
  # before Kitsune. libomp.so is also present in a directory that is not easily
  # accessible, so the library cannot be copied to the correct "staging"
  # directory where it can be found when linking the runtime.
  message(STATUS "Kitsune: openmp target enabled")

  set(KITSUNE_OPENMP_BUILD_CONFIGURE_FLAGS ""
    CACHE STRING
    "Additional options to be passed to CMake when configuring the openmp
    runtime. These are only used when building openmp as part of building
    Kitsune itself")

  set(KITSUNE_OPENMP_BUILD_COMPILE_FLAGS ""
    CACHE STRING
    "Additional compiler flags needed to build the openmp runtime. These are
    only used when building Cheetah as part of building Kitsune itself")

  set(KITSUNE_OPENMP_BUILD_LINK_FLAGS ""
    CACHE STRING
    "Additional linker flags needed to build the openmp runtime. These are only
    used when building Cheetah as part of building Kitsune itself")

  prepare_extproj_cmdline_opts("${KITSUNE_OPENMP_BUILD_COMPILE_FLAGS}"
    KITSUNE_OPENMP_C_FLAGS)
  prepare_extproj_cmdline_opts("${KITSUNE_OPENMP_BUILD_COMPILE_FLAGS}"
    KITSUNE_OPENMP_CXX_FLAGS)
  prepare_extproj_cmdline_opts("${KITSUNE_OPENMP_BUILD_LINK_FLAGS}"
    KITSUNE_OPENMP_LINKER_FLAGS)

  set(KITSUNE_OPENMP_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../runtimes)
  set(KITSUNE_OPENMP_BINARY_DIR ${KITSUNE_TARGETS_BINARY_DIR}/openmp/build)
  set(KITSUNE_OPENMP_INSTALL_DIR ${KITSUNE_TARGETS_BINARY_DIR}/openmp/install)

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
  set(KITSUNE_OPENMP_BUILD_CMAKE_FLAGS
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_C_FLAGS=${KITSUNE_OPENMP_C_FLAGS}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_CXX_FLAGS=${KITSUNE_OPENMP_CXX_FLAGS}
    -DCMAKE_EXE_LINKER_FLAGS=${KITSUNE_OPENMP_LINKER_FLAGS}
    -DCMAKE_SHARED_LINKER_FLAGS=${KITSUNE_OPENMP_LINKER_FLAGS}
    -DCMAKE_INSTALL_PREFIX=${KITSUNE_OPENMP_INSTALL_DIR}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_COLOR_DIAGNOSTICS=${CMAKE_COLOR_DIAGNOSTICS}
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -DCLANG_RESOURCE_DIR=${CLANG_RESOURCE_INTDIR}
    -DLLVM_BINARY_DIR=${CMAKE_BINARY_DIR}
    -DLLVM_ENABLE_RUNTIMES=openmp
    -DLLVM_TARGET_ARCH=${LLVM_TARGET_ARCH}
    -DLLVM_HOST_TRIPLE=${LLVM_HOST_TRIPLE}
    -DOPENMP_STANDALONE_BUILD=ON
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF
    -DOPENMP_ENABLE_LIBOMP_PROFILING=OFF
    -DOPENMP_ENABLE_OMPT_TOOLS=OFF
    -DLIBOMP_OMPD_SUPPORT=OFF)

  # The openmp tests are not very cheap, so we don't run them before installing.
  # But we probably should.
  ExternalProject_Add(openmp
    SOURCE_DIR              ${KITSUNE_OPENMP_SOURCE_DIR}
    BINARY_DIR              ${KITSUNE_OPENMP_BINARY_DIR}
    INSTALL_DIR             ${CLANG_RESOURCE_INTDIR}
    CMAKE_ARGS              ${KITSUNE_OPENMP_BUILD_CMAKE_FLAGS}
    LIST_SEPARATOR          |
    TEST_BEFORE_INSTALL     FALSE
    USES_TERMINAL_PATCH     TRUE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD     TRUE
    USES_TERMINAL_INSTALL   TRUE
    UPDATE_DISCONNECTED     TRUE
    EXCLUDE_FROM_ALL        TRUE)

  # This is the only library that we care about because it has to be linked to
  # Kitsune's runtime.
  set(LIBNAME_OMP ${SHLIB_PFX}omp${SHLIB_SFX})
  set(KITSUNE_OPENMP_LIB_OMP ${CLANG_RESOURCE_INT_LIBDIR}/${LIBNAME_OMP})

  set(KITSUNE_OPENMP_INSTALL_LIBDIR
    ${KITSUNE_OPENMP_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR})

  # "Install" the required OpenMP files to the top-level build directory. This
  # allows Kitsune to be run from the build directory. The only output we care
  # about is the main OpenMP library, libomp. But we copy the whole directory
  # because that main library may be a symlink.
  add_custom_command(
    OUTPUT ${KITSUNE_OPENMP_LIB_OMP}
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${KITSUNE_OPENMP_INSTALL_LIBDIR}
    ${CLANG_RESOURCE_INT_LIBDIR}
    DEPENDS openmp
    COMMENT "Copying OpenMP to the top-level build directory"
    VERBATIM USES_TERMINAL)

  # This target drives the command to "install" OpenMP to the build directory.
  add_custom_target(openmp-copy-to-build-dir
    DEPENDS ${KITSUNE_OPENMP_LIB_OMP})

  # Add this to the dependencies for Kitsune. This will ensure that OpenMP gets
  # built. We have excluded it from ALL, so it won't be built automatically.
  list(APPEND KITRT_DEPENDS openmp-copy-to-build-dir)

  # When installing Kitsune, make sure to install libomp as well. We don't
  # install the whole directory because we only need the libraries.
  # NOTE: The trailing slash after the directory to copy is required.
  install(DIRECTORY ${KITSUNE_OPENMP_INSTALL_LIBDIR}/
    DESTINATION ${CLANG_RESOURCE_INSTALL_LIBDIR}
    USE_SOURCE_PERMISSIONS)

  set(KITSUNE_OPENMP_INCDIRS
    ${LLVM_MONOREPO_SOURCE_DIR}/openmp/runtime/src
    ${KITSUNE_OPENMP_BINARY_DIR}/openmp/runtime/src)
  set(KITSUNE_OPENMP_LIBS ${KITSUNE_OPENMP_LIB_OMP})

  string(REPLACE ";" "|" KITSUNE_OPENMP_INCDIRS "${KITSUNE_OPENMP_INCDIRS}")
  string(REPLACE ";" "|" KITSUNE_OPENMP_LIBS "${KITSUNE_OPENMP_LIBS}")
endif ()

if (KITSUNE_QTHREADS_ENABLED)
  # We currently don't allow using a pre-built qthreads. It should be ok to do
  # this since it is not likely that the interface, or behavior, will change.
  # Since this library is unlikely to be present on the average system on which
  # we expect Kitsune to be used, we are not particularly keen on supporting
  # this anytime soon.
  message(STATUS "Kitsune: qthreads target enabled")

  set(KITSUNE_QTHREADS_BUILD_CONFIGURE_FLAGS ""
    CACHE STRING
    "Additional options to be passed to CMake when configuring qthreads. These
    are only used when building qthreads as part of building Kitsune itself")

  set(KITSUNE_QTHREADS_BUILD_COMPILE_FLAGS ""
    CACHE STRING
    "Additional compiler flags needed to build qthreads. These are only used
    when building qthreads as part of building Kitsune itself")

  set(KITSUNE_QTHREADS_BUILD_LINK_FLAGS ""
    CACHE STRING
    "Additional linker flags needed to build qthreads. These are only used when
    building qthreads as part of building Kitsune itself")

  prepare_extproj_cmdline_opts("${KITSUNE_QTHREADS_BUILD_COMPILE_FLAGS}"
    KITSUNE_QTHREADS_C_FLAGS)
  prepare_extproj_cmdline_opts("${KITSUNE_QTHREADS_BUILD_COMPILE_FLAGS}"
    KITSUNE_QTHREADS_CXX_FLAGS)
  prepare_extproj_cmdline_opts("${KITSUNE_QTHREADS_BUILD_LINK_FLAGS}"
    KITSUNE_QTHREADS_LINKER_FLAGS)

  set(KITSUNE_QTHREADS_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
  set(KITSUNE_QTHREADS_SOURCE_DIR ${KITSUNE_TARGETS_BINARY_DIR}/qthreads/qthreads)
  set(KITSUNE_QTHREADS_BINARY_DIR ${KITSUNE_TARGETS_BINARY_DIR}/qthreads/build)
  set(KITSUNE_QTHREADS_INSTALL_DIR ${KITSUNE_TARGETS_BINARY_DIR}/qthreads/install)
  set(KITSUNE_QTHREADS_BUILD_CMAKE_FLAGS
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
    -DCMAKE_INSTALL_PREFIX=${KITSUNE_QTHREADS_INSTALL_DIR}
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -DQTHREADS_TOPOLOGY=hwloc
    -DQTHREADS_SCHEDULER=distrib)

  ExternalProject_Add(qthreads
    GIT_REPOSITORY          https://github.com/sandialabs/qthreads.git
    GIT_TAG                 1.23
    SOURCE_DIR              ${KITSUNE_QTHREADS_SOURCE_DIR}
    BINARY_DIR              ${KITSUNE_QTHREADS_BINARY_DIR}
    INSTALL_DIR             ${KITSUNE_QTHREADS_INSTALL_DIR}
    CMAKE_ARGS              ${KITSUNE_QTHREADS_BUILD_CMAKE_FLAGS}
    LIST_SEPARATOR          |
    TEST_BEFORE_INSTALL     TRUE
    USES_TERMINAL_PATCH     TRUE
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD     TRUE
    USES_TERMINAL_INSTALL   TRUE
    UPDATE_DISCONNECTED     TRUE
    EXCLUDE_FROM_ALL        TRUE)

  # This is the only library that we care about because it has to be linked to
  # Kitsune's runtime.
  set(LIBNAME_QTHREAD ${SHLIB_PFX}qthread${SHLIB_SFX})
  set(KITSUNE_QTHREADS_LIB_QTHREAD
    ${CLANG_RESOURCE_INT_LIBDIR}/${LIBNAME_QTHREAD})

  set(KITSUNE_QTHREADS_INSTALL_LIBDIR
    ${KITSUNE_QTHREADS_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR})

  # "Install" the required Qthreads files to the top-level build directory. This
  # allows Kitsune to be run from the build directory. The only output we care
  # about is the main Qthreads library, libqthread. But we copy the whole
  # directory because that main library may be a symlink.
  add_custom_command(
    OUTPUT ${KITSUNE_QTHREADS_LIB_QTHREAD}
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
    ${KITSUNE_QTHREADS_INSTALL_LIBDIR}
    ${CLANG_RESOURCE_INT_LIBDIR}
    DEPENDS qthreads
    COMMENT "Copying Qthreads to the top-level build directory"
    VERBATIM USES_TERMINAL)

  # This target drives the command to "install" qthreads to the build directory.
  add_custom_target(qthreads-copy-to-build-dir
    DEPENDS ${KITSUNE_QTHREADS_LIB_QTHREAD})

  # Add this to the dependencies for Kitsune. This will ensure that qthreads
  # gets built. We have excluded it from ALL, so it won't be built
  # automatically.
  list(APPEND KITRT_DEPENDS qthreads-copy-to-build-dir)

  # When installing Kitsune, make sure to install libqthread as well. We don't
  # install the whole directory because we only need the libraries.
  # NOTE: The trailing slash after the directory to copy is required.
  install(DIRECTORY ${KITSUNE_QTHREADS_INSTALL_LIBDIR}/
    DESTINATION ${CLANG_RESOURCE_INSTALL_LIBDIR}
    USE_SOURCE_PERMISSIONS
    PATTERN "cmake" EXCLUDE)

  set(KITSUNE_QTHREADS_INCDIRS ${KITSUNE_QTHREADS_INSTALL_DIR}/include)
  set(KITSUNE_QTHREADS_LIBS ${KITSUNE_QTHREADS_LIB_QTHREAD})

  string(REPLACE ";" "|" KITSUNE_QTHREADS_INCDIRS "${KITSUNE_QTHREADS_INCDIRS}")
  string(REPLACE ";" "|" KITSUNE_QTHREADS_LIBS "${KITSUNE_QTHREADS_LIBS}")
endif()

if (KITSUNE_REALM_ENABLED)
  message(FATAL_ERROR "NOT YET IMPLEMENTED: 'realm' tapir target")
  message(STATUS "Kitsune: realm tapir target enabled")

  # FIXME: Do find_package or ExternalProject_add or something else here for
  # realm. Odds are that we will end up building realm like we build Cheetah.
  # In any case, this message is here to ensure that when (if) we ever resurrect
  # the realm tapir target, we don't forget to do the right thing here.
  message(FATAL_ERROR "Finding/building Realm has not been implemented")
endif()

# The KITSUNE_COMMON_LIB_* variables are passed to the configure command for
# Kitsune's runtime below. If we pass the list as-is, it nearly always fails
# because the terminal interprets the semicolon in the string-representation of
# a cmake list as a terminator in the shell command. To work around this, we
# use a pipe '|' as a separator (since it is unlikely to appear in library or
# path names).
#
# The same technique is used when determining the library names and install
# directories of Kitsune's dependencies that are configured/built below. In
# the corresponding External_Project commands, the LIST_SEPARATOR option is used
# to inform cmake that the pipe symbol is actually a list separator.
#
populate_unique_parents(KITSUNE_COMMON_LIBS KITSUNE_COMMON_LIBDIRS)
string(REPLACE ";" "|" KITSUNE_COMMON_LIB_NAMES "${KITSUNE_COMMON_LIB_NAMES}")
string(REPLACE ";" "|" KITSUNE_COMMON_LIBS "${KITSUNE_COMMON_LIBS}")
string(REPLACE ";" "|" KITSUNE_COMMON_LIBDIRS "${KITSUNE_COMMON_LIBDIRS}")

set(KITRT_BUILD_CONFIGURE_FLAGS ""
  CACHE STRING
  "Additional options to be passed to CMake when configuring Kitsune's runtime.
  These are only used when building the runtime as part of building Kitsune
  itself")

set(KITRT_BUILD_COMPILE_FLAGS ""
  CACHE STRING
  "Additional compiler flags needed to build Kitsune's runtime. These are only
  used when building Kitsune's runtime as part of building Kitsune itself")

set(KITRT_BUILD_LINK_FLAGS ""
  CACHE STRING
  "Additional linker flags needed to build Kitsune's runtime. These are only
  used when building Kitsune's runtime as part of building Kitsune itself")

prepare_extproj_cmdline_opts("${KITRT_BUILD_COMPILE_FLAGS}"
  KITRT_C_FLAGS)
prepare_extproj_cmdline_opts("${KITRT_BUILD_COMPILE_FLAGS}"
  KITRT_CXX_FLAGS)
prepare_extproj_cmdline_opts("${KITRT_BUILD_LINK_FLAGS}"
  KITRT_LINKER_FLAGS)

# Because we treat kitsune's runtime as an external project, we handle it the
# same way we do the tapir targets i.e., we create an "install" directory
# within the build directory into which all the files will be "installed" at
# build time with a separate install command to copy these out to the actual
# install directory for kitsune as specified by ${CMAKE_INSTALL_PREFIX}.
set(KITRT_SOURCE_DIR ${KITSUNE_SOURCE_DIR}/runtime)
set(KITRT_BINARY_DIR ${KITSUNE_BINARY_DIR}/runtime/build)

set(KITRT_CONFIGURE_ARGS
  -DCMAKE_C_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang
  -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
  -DCMAKE_C_COMPILER_WORKS=1
  -DCMAKE_C_FLAGS=${KITRT_C_FLAGS}
  -DCMAKE_CXX_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++
  -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
  -DCMAKE_CXX_COMPILER_WORKS=1
  -DCMAKE_CXX_FLAGS=${KITRT_CXX_FLAGS}
  -DCMAKE_EXE_LINKER_FLAGS=${KITRT_LINKER_FLAGS}
  -DCMAKE_SHARED_LINKER_FLAGS=${KITRT_LINKER_FLAGS}
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_COLOR_DIAGNOSTICS=${CMAKE_COLOR_DIAGNOSTICS}
  -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DCLANG_RESOURCE_INTDIR=${CLANG_RESOURCE_INTDIR}
  -DCLANG_RESOURCE_INSTALL_DIR=${CLANG_RESOURCE_INSTALL_DIR}
  -DLLVM_CONFIG_PATH=${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-config
  -DLLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR}
  -DLLVM_VERSION_MINOR=${LLVM_VERSION_MINOR}
  -DLLVM_VERSION_PATCH=${LLVM_VERSION_PATCH}
  -DLLVM_BINARY_DIR=${LLVM_BINARY_DIR}
  -DLLVM_INCLUDE_DIR=${LLVM_INCLUDE_DIR}
  -DLLVM_LIBRARY_DIR=${LLVM_LIBRARY_DIR}
  -DLLVM_LIBRARY_OUTPUT_INTDIR=${LLVM_LIBRARY_OUTPUT_INTDIR}
  -DLLVM_RUNTIME_OUTPUT_INTDIR=${LLVM_RUNTIME_OUTPUT_INTDIR}
  -DLLVM_LIBDIR_SUFFIX=${LLVM_LIBDIR_SUFFIX}
  -DLLVM_MAIN_INCLUDE_DIR=${LLVM_MAIN_INCLUDE_DIR}
  -DLLVM_RUNTIME_OUTPUT_INTDIR=${LLVM_RUNTIME_OUTPUT_INTDIR}
  -DLLVM_ENABLE_ASSERTIONS=${LLVM_ENABLE_ASSERTIONS}
  -DLLVM_DEFAULT_TARGET_TRIPLE=${LLVM_DEFAULT_TARGET_TRIPLE}
  -DKITSUNE_SOURCE_DIR=${KITSUNE_SOURCE_DIR}
  -DKITSUNE_BINARY_DIR=${KITSUNE_BINARY_DIR}
  -DKITSUNE_COMMON_LIB_NAMES=${KITSUNE_COMMON_LIB_NAMES}
  -DKITSUNE_COMMON_LIBDIRS=${KITSUNE_COMMON_LIBDIRS}
  -DKITSUNE_COMMON_LIBS=${KITSUNE_COMMON_LIBS}
  -DKITRT=${KITRT}
  -DKITRT_STATIC=${KITRT_STATIC})

list(APPEND KITRT_CONFIGURE_ARGS
  -DKITSUNE_OPENMP_INCDIRS=${KITSUNE_OPENMP_INCDIRS}
  -DKITSUNE_OPENMP_LIBS=${KITSUNE_OPENMP_LIBS})

list(APPEND KITRT_CONFIGURE_ARGS
  -DKITSUNE_PAPI_ENABLED=${KITSUNE_PAPI_ENABLED}
  -DKITSUNE_CUDA_ENABLED=${KITSUNE_CUDA_ENABLED}
  -DKITSUNE_HIP_ENABLED=${KITSUNE_HIP_ENABLED}
  -DKITSUNE_OPENCILK_ENABLED=${KITSUNE_OPENCILK_ENABLED}
  -DKITSUNE_QTHREADS_ENABLED=${KITSUNE_QTHREADS_ENABLED})

if (KITSUNE_PAPI_ENABLED)
  list(APPEND KITRT_CONFIGURE_ARGS
    -DKITSUNE_PAPI_INCDIRS=${KITSUNE_PAPI_INCDIRS}
    -DKITSUNE_PAPI_LIBS=${KITSUNE_PAPI_LIBS})
endif ()

if (KITSUNE_CUDA_ENABLED)
  list(APPEND KITRT_CONFIGURE_ARGS
    -DKITSUNE_CUDA_DEFNS=${KITSUNE_CUDA_DEFNS}
    -DKITSUNE_CUDA_INCDIRS=${KITSUNE_CUDA_INCDIRS}
    -DKITSUNE_CUDA_LIB_NAMES=${KITSUNE_CUDA_LIB_NAMES}
    -DKITSUNE_CUDA_LIBDIRS=${KITSUNE_CUDA_LIBDIRS}
    -DKITSUNE_CUDA_LIBS=${KITSUNE_CUDA_LIBS})
endif()

if (KITSUNE_HIP_ENABLED)
  list(APPEND KITRT_CONFIGURE_ARGS
    -DKITSUNE_HIP_INCDIRS=${KITSUNE_HIP_INCDIRS}
    -DKITSUNE_HIP_LIB_NAMES=${KITSUNE_HIP_LIB_NAMES}
    -DKITSUNE_HIP_LIBDIRS=${KITSUNE_HIP_LIBDIRS}
    -DKITSUNE_HIP_LIBS=${KITSUNE_HIP_LIBS})
endif()

if (KITSUNE_OPENCILK_ENABLED)
  list(APPEND KITRT_CONFIGURE_ARGS
    -DKITSUNE_OPENCILK_LIBS=${KITSUNE_OPENCILK_LIBS})
endif ()

if (KITSUNE_QTHREADS_ENABLED)
  list(APPEND KITRT_CONFIGURE_ARGS
    -DKITSUNE_QTHREADS_INCDIRS=${KITSUNE_QTHREADS_INCDIRS}
    -DKITSUNE_QTHREADS_LIBS=${KITSUNE_QTHREADS_LIBS})
endif ()

# We don't descend into the runtime subdirectory because that should only be
# configured when the runtimes are built. All the cmake environment variables
# that the runtime might need have to be passed through here. BUILD_ALWAYS must
# be set to TRUE here, otherwise runtime/ will not be checked for changes, which
# is not terribly helpful for Kitsune developers.
#
# We don't install this because we have complete control over the build process,
# so we know exactly what artifacts are built and where. So we just get hold
# of them directly from the build directory.
ExternalProject_Add(${KITRT}
  DEPENDS                    ${KITRT_DEPENDS}
  SOURCE_DIR                 ${KITRT_SOURCE_DIR}
  BINARY_DIR                 ${KITRT_BINARY_DIR}
  CMAKE_ARGS                 ${KITRT_CONFIGURE_ARGS}
  INSTALL_COMMAND            ""
  LIST_SEPARATOR             |
  USES_TERMINAL_CONFIGURE    TRUE
  USES_TERMINAL_BUILD        TRUE
  BUILD_ALWAYS               TRUE
  CONFIGURE_HANDLED_BY_BUILD TRUE)

# Copy the kitsune runtime files into a "staging" area. This allows kitsune to
# be run from the build directory. This must be `ALL`, otherwise it will not be
# run since nothing depends on it.
add_custom_command(TARGET ${KITRT} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${KITRT_BINARY_DIR}/${LIBNAME_KITRT}
  ${KITRT_BINARY_DIR}/${LIBNAME_KITRT_STATIC}
  ${CLANG_RESOURCE_INT_LIBDIR}
  COMMENT "Install ${KITRT} to the top-level build directory"
  VERBATIM USES_TERMINAL)

install(PROGRAMS
  ${KITRT_BINARY_DIR}/${LIBNAME_KITRT}
  ${KITRT_BINARY_DIR}/${LIBNAME_KITRT_STATIC}
  DESTINATION ${CLANG_RESOURCE_INSTALL_LIBDIR})

# We do not enter the lib subdirectory here. The contents of lib/ are treated as
# core LLVM libraries which become part of libLLVM.so. Therefore, the contents
# are handled in llvm/lib/CMakeLists.txt.
add_subdirectory(cmake/modules)
add_subdirectory(include/kitsune)
add_subdirectory(tools)

if (KITSUNE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif ()

if (KITSUNE_ENABLE_SPHINX OR KITSUNE_ENABLE_DOXYGEN)
  add_subdirectory(docs)
endif ()

if(LLVM_INCLUDE_TESTS AND KITSUNE_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(unittests)
endif()

if (KITSUNE_ENABLE_EXPERIMENTS)
  add_subdirectory(experiments)
endif ()

# Create the rules to create a "base" Kitsune executable. This is simply a copy
# of the base clang (or flang) executable. We have to create a copy because a
# symlink will not work as we want. Very early in the process, the symlinks are
# stripped away and the underlying executable is exposed to the driver. We need
# this to *not* be named clang in order to determine whether we should be
# compiling in "kitsune mode" (this changes the default behavior for certain
# things such as disabling exceptions, setting the fp-contract option to a
# different value etc. so we have to get this right). To be consistent with
# clang, we do not create a separate frontend for C++.
set(KITCC_MAIN_FILENAME "${KITSUNE_C_FRONTEND}-${LLVM_VERSION_MAJOR}")
set(KITCC_MAIN ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${KITCC_MAIN_FILENAME})
add_custom_command(
  OUTPUT ${KITCC_MAIN}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:clang> ${KITCC_MAIN}
  DEPENDS clang
  COMMENT "Creating ${KITCC_MAIN_FILENAME}"
  VERBATIM USES_TERMINAL)

# Create a symlink kitcc -> kitcc-${LLVM_VERSION_MAJOR}
set(KITCC ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${KITSUNE_C_FRONTEND})
add_custom_command(
    OUTPUT ${KITCC}
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${KITCC_MAIN_FILENAME} ${KITCC}
    DEPENDS ${KITCC_MAIN}
    COMMENT "Creating symlink ${KITSUNE_C_FRONTEND} to ${KITCC_MAIN_FILENAME}"
    VERBATIM USES_TERMINAL)
add_custom_target(kitcc ALL DEPENDS ${KITCC})

# Create a symlink kit++ -> kitcc-${LLVM_VERSION_MAJOR}
set(KITXX ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${KITSUNE_CXX_FRONTEND})
add_custom_command(
  OUTPUT ${KITXX}
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${KITCC_MAIN_FILENAME} ${KITXX}
  DEPENDS ${KITCC_MAIN}
  COMMENT "Creating symlink ${KITSUNE_CXX_FRONTEND} to ${KITCC_MAIN_FILENAME}"
  VERBATIM USES_TERMINAL)
add_custom_target(kit++ ALL DEPENDS ${KITXX})

install(PROGRAMS ${KITCC_MAIN} ${KITCC} ${KITXX}
  DESTINATION ${CMAKE_INSTALL_BINDIR})

if (KITSUNE_Fortran_ENABLED)
  # Copy flang to kitfc-${LLVM_VERSION_MAJOR}
  set(KITFC_MAIN_FILENAME "${KITSUNE_Fortran_FRONTEND}-${LLVM_VERSION_MAJOR}")
  set(KITFC_MAIN ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${KITFC_MAIN_FILENAME})
  add_custom_command(
    OUTPUT ${KITFC}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:flang> ${KITFC_MAIN}
    DEPENDS flang
    COMMENT "Creating ${KITFC_MAIN_FILENAME}"
    VERBATIM USES_TERMINAL)

  # Create a symlink kitfc -> kitfc-${LLVM_VERSION_MAJOR}
  set(KITFC ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${KITSUNE_Fortran_FRONTEND})
  add_custom_command(
    OUTPUT ${KITFC}
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${KITFC_MAIN_FILENAME} ${KITFC}
    DEPENDS ${KITFC_MAIN}
    COMMENT "Creating symlink ${KITSUNE_Fortran_FRONTEND} to ${KITFC_MAIN_FILENAME}"
    VERBATIM USES_TERMINAL)
  add_custom_target(kitfc ALL DEPENDS ${KITFC})

  install(PROGRAMS ${KITFC_MAIN} ${KITFC}
    DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
