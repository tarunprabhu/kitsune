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
find_library(LIB_LLVM
  NAMES LLVM
  LLVM-${LLVM_VERSION_MAJOR}
  LLVM-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}
  REQUIRED
  PATHS ${LLVM_LIBRARY_OUTPUT_INTDIR}
  NO_DEFAULT_PATH)

# ${KITSUNE_RUNTIME_LIBNAME} is passed in by cmake when the runtime is
# configured.
set(KITRT ${KITSUNE_RUNTIME_LIBNAME})

add_library(${KITRT} SHARED
  kitrt.cpp
  debug.cpp
  memory.cpp
  memory_map.cpp)

target_include_directories(${KITRT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${KITRT} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CLANG_RESOURCE_INTDIR}/lib)

target_include_directories(${KITRT}
  BEFORE PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
  ${LLVM_INCLUDE_DIR}
  ${KITSUNE_SOURCE_DIR}/include
  ${KITSUNE_RUNTIME_INCLUDE_DIR})

# FIXME: We probably don't need to explicitly link pthread and rt. These are
# only necessary for older versions of glibc, but we might get some antique
# software stacks on HPC machines.
target_link_libraries(${KITRT}
  PUBLIC
  ${LIB_LLVM}
  pthread
  rt
  dl)

# The KITSUNE_CUDA_* variables are defined in kitsune/CMakeLists.txt. They are
# passed in via CMake because the Kitsune runtime is treated as an external
# project.
if (KITSUNE_CUDA_ENABLED)
  target_sources(${KITRT} PUBLIC
    cuda/kitcuda.cpp
    cuda/dylib_support.cpp
    cuda/launching.cpp
    cuda/memory.cpp
    cuda/streams.cpp)

  target_compile_definitions(${KITRT} PUBLIC KITRT_CUDA_ENABLED)
  target_include_directories(${KITRT} SYSTEM PUBLIC ${KITSUNE_CUDA_INCLUDE_DIR})
  target_include_directories(${KITRT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)

  target_link_directories(${KITRT} PUBLIC
    ${KITSUNE_CUDA_LIBRARY_DIR})

  target_link_libraries(${KITRT} PUBLIC
    ${KITSUNE_CUDA_LIB_NVPTX_STATIC}
    ${KITSUNE_CUDA_LIB_CUDART}
    ${KITSUNE_CUDA_LIB_CUDA})

  if (KITCUDA_ENABLE_NVTX)
    target_compile_definitions(${KITRT} PUBLIC KITCUDA_ENABLE_NVTX)

    find_library(LIB_NVTOOLS_EXT nvToolsExt REQUIRED
      PATHS ${KITSUNE_CUDA_LIBRARY_DIR}
      NO_DEFAULT_PATH)

    target_link_libraries(${KITRT} PUBLIC ${LIB_NVTOOLS_EXT})
  endif()
endif()

# The KITSUNE_HIP_* variables are defined in kitsune/CMakeLists.txt. They are
# passed in via CMake because the Kitsune runtime is treated as an external
# project.
#
# Note on AMD HIP CMake details...  When is HIP not HIP?  When it is HIP...
# It is easy to confuse requirements for HIP as the HIP "language" vs. the
# HIP runtime.   The language has tangled dependencies within a LLVM context
# as they build around LLVM and Clang too.  Using some of the 'hip::host'
# targets are actually intended for the language and not runtime libraries.
# Unfortunately, these details seem to be a moving target with ecah release
# of rocm/hip...
if (KITSUNE_HIP_ENABLED)
  target_sources(${KITRT} PUBLIC
    hip/kithip.cpp
    hip/dylib_support.cpp
    hip/launching.cpp
    hip/memory.cpp
    hip/streams.cpp)

  target_compile_definitions(${KITRT} PUBLIC KITRT_HIP_ENABLED)
  target_compile_definitions(${KITRT} PUBLIC __HIP_PLATFORM_AMD__)
  target_include_directories(${KITRT} PUBLIC ${KITSUNE_HIP_INCLUDE_DIR})
  target_link_directories(${KITRT} PUBLIC ${KITSUNE_HIP_LIBRARY_DIR})
  target_link_libraries(${KITRT} PUBLIC ${KITSUNE_HIP_LIB_AMDHIP64})
endif()

# $ORIGIN here is the location of libkitrt.so. This will be installed to clang's
# resource directory. Since libkitrt links against libLLVM.so, we have to add an
# rpath to the "top-level" lib directory where libLLVM can be found. $ORIGIN is
# a subdirectory of this top-level directory. It is usually:
#
#    <top-level-lib-dir>/clang/19/lib
#
set_property(TARGET ${KITRT} APPEND PROPERTY
  INSTALL_RPATH "$ORIGIN/../../../")
set_property(TARGET ${KITRT} APPEND PROPERTY
  BUILD_RPATH "$ORIGIN/../../../")

install(TARGETS ${KITRT}
  DESTINATION ${CLANG_RESOURCE_SUBDIR}/lib)
