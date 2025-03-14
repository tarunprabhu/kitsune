# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
#
cmake_minimum_required(VERSION 3.20.0)

project(kitrt C CXX)

# FIXME: Should this be named KITRT_* since we don't have a separate kitcuda
# library any longer?
option(KITCUDA_ENABLE_NVTX "Enable NVTX profiling within the runtime" OFF)

# Since this is built as an external project, we have to explicitly find
# libLLVM. This is guaranteed to have been built, but we need to be careful that
# we don't pick up anything from the system. ${LLVM_LIBRARY_OUTPUT_INTDIR} will
# have been passed in by cmake.
#
# NOTE: We have removed the dependence on libLLVM for now, but we keep this
# here for now. There is a lot of unused code in the runtime, most of which
# should probably be deleted but is still kept around in case we decide to
# resurrect something. If we do, some of it might require libLLVM.
#
find_library(LIB_LLVM
  NAMES LLVM
  LLVM-${LLVM_VERSION_MAJOR}
  LLVM-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}
  REQUIRED
  PATHS ${LLVM_LIBRARY_OUTPUT_INTDIR}
  NO_DEFAULT_PATH)

# ${KITSUNE_LIBNAME} is passed in by cmake when the runtime is configured.
set(KITSUNE_OBJLIB ${KITSUNE_LIBNAME}_objlib)

add_library(${KITSUNE_OBJLIB} OBJECT
  kitrt.cpp
  memory.cpp
  memory_map.cpp)

set_target_properties(${KITSUNE_OBJLIB} PROPERTIES
  POSITION_INDEPENDENT_CODE 1)

target_include_directories(${KITSUNE_OBJLIB} PUBLIC
  ${LLVM_MAIN_INCLUDE_DIR}
  ${LLVM_INCLUDE_DIR}
  ${KITSUNE_SOURCE_DIR}/include
  ${KITSUNE_RUNTIME_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})

# FIXME: We probably don't need to explicitly link pthread and rt. These are
# only necessary for older versions of glibc, but we might get some antique
# software stacks on HPC machines. Some details are here:
#
# https://developers.redhat.com/articles/2021/12/17/why-glibc-234-removed-libpthread
#
target_link_libraries(${KITSUNE_OBJLIB} PUBLIC
  pthread rt dl)

# The KITSUNE_CUDA_* variables are defined in kitsune/CMakeLists.txt. They are
# passed in via CMake because the Kitsune runtime is treated as an external
# project.
if (KITSUNE_CUDA_ENABLED)
  target_sources(${KITSUNE_OBJLIB} PUBLIC
    cuda/kitcuda.cpp
    cuda/dylib_support.cpp
    cuda/launching.cpp
    cuda/memory.cpp
    cuda/streams.cpp)

  # FIXME: The only use of KITRT_CUDA_ENABLED seem to be some runtime
  # initialization code that is probably obsolete. The runtime is probably due
  # for some drastic cleanup.
  target_compile_definitions(${KITSUNE_OBJLIB} PUBLIC
    KITRT_CUDA_ENABLED)

  target_include_directories(${KITSUNE_OBJLIB} SYSTEM PUBLIC
    ${KITSUNE_CUDA_INCLUDE_DIR})

  target_include_directories(${KITSUNE_OBJLIB} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda)

  target_link_directories(${KITSUNE_OBJLIB} PUBLIC
    ${KITSUNE_CUDA_LIBRARY_DIR})

  target_link_libraries(${KITSUNE_OBJLIB} PUBLIC
    ${KITSUNE_CUDA_LIB_NVPTX_STATIC}
    ${KITSUNE_CUDA_LIB_CUDART_STATIC}
    ${KITSUNE_CUDA_LIB_CUDA})

  if (KITCUDA_ENABLE_NVTX)
    find_library(LIB_NVTOOLS_EXT nvToolsExt REQUIRED
      PATHS ${KITSUNE_CUDA_LIBRARY_DIR}
      NO_DEFAULT_PATH)

    target_compile_definitions(${KITSUNE_OBJLIB} PUBLIC
      KITCUDA_ENABLE_NVTX)

    target_link_libraries(${KITSUNE_OBJLIB} PUBLIC
      ${LIB_NVTOOLS_EXT})
  endif()
endif()

# The KITSUNE_HIP_* variables are defined in kitsune/CMakeLists.txt. They are
# passed in via CMake because the Kitsune runtime is treated as an external
# project.
#
# Note on AMD HIP CMake details...  When is HIP not HIP?  When it is HIP...
# It is easy to confuse requirements for HIP as the HIP "language" vs. the
# HIP runtime. Unfortunately, these details seem to be a moving target with
# each release of rocm/hip.
if (KITSUNE_HIP_ENABLED)
  target_sources(${KITSUNE_OBJLIB} PUBLIC
    hip/kithip.cpp
    hip/dylib_support.cpp
    hip/launching.cpp
    hip/memory.cpp
    hip/streams.cpp)

  target_compile_definitions(${KITSUNE_OBJLIB} PUBLIC
    # FIXME: The only use of KITRT_HIP_ENABLED seem to be some runtime
    # initialization code that is probably obsolete. The runtime is due for some
    # drastic cleanup.
    KITRT_HIP_ENABLED

    # FIXME: It looks like __HIP_PLATFORM_AMD__ is defined in the kitsune
    # runtime headers for hip. This is probably unnecessary.
    __HIP_PLATFORM_AMD__)

  target_include_directories(${KITSUNE_OBJLIB} PUBLIC
    ${KITSUNE_HIP_INCLUDE_DIR})

  target_link_directories(${KITSUNE_OBJLIB} PUBLIC
    ${KITSUNE_HIP_LIBRARY_DIR})

  target_link_libraries(${KITSUNE_OBJLIB} PUBLIC
    ${KITSUNE_HIP_LIB_AMDHIP64})
endif()

# Create the static library, libkitrt_static.a
add_library(${KITSUNE_LIBNAME_STATIC} STATIC
  $<TARGET_OBJECTS:${KITSUNE_OBJLIB}>)

set_target_properties(${KITSUNE_LIBNAME_STATIC} PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CLANG_RESOURCE_INTDIR}/lib
  LINK_LIBRARIES $<LINK_LIBRARY:${KITSUNE_OBJLIB}>)

# Create the shared library libkitrt.so
add_library(${KITSUNE_LIBNAME} SHARED
  $<TARGET_OBJECTS:${KITSUNE_OBJLIB}>)

set_target_properties(${KITSUNE_LIBNAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CLANG_RESOURCE_INTDIR}/lib
  LINK_LIBRARIES $<LINK_LIBRARY:${KITSUNE_OBJLIB}>)

# $ORIGIN is the location of libkitrt.so. This is
#
#     <prefix>/lib/clang/${LLVM_VERSION_MAJOR}/lib
#
# where <prefix> is either the build or the install prefix. Right now, there is
# no dependence on libLLVM.so, but in case we have to add an rpath to the
# top-level lib directory, i.e. <prefix>/lib.
#
set_property(TARGET ${KITSUNE_LIBNAME} APPEND
  PROPERTY BUILD_RPATH "$ORIGIN/../../../"
  PROPERTY INSTALL_RPATH "$ORIGIN/../../../")

install(TARGETS ${KITSUNE_LIBNAME} ${KITSUNE_LIBNAME_STATIC}
  DESTINATION ${CLANG_RESOURCE_SUBDIR}/lib)
