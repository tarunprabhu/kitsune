# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
#
cmake_minimum_required(VERSION 3.20.0)

project(kitrt C CXX)

include(FetchContent)

set(KITRT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KITRT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(KITRT_INCLUDE_DIR ${KITSUNE_RUNTIME_DIR})

# There is support for two different levels of debugging.  The general
# debug mode (KITRT_ENABLE_DEBUG) will enable runtime assertion checks
# that will cause the runtime to both check and abort on any error
# conditions from the underlying runtimes (e.g., CUDA, HIP, etc.).  The
# second option (KITRT_ENABLE_VERBOSE) will report runtime details and
# give a general feel for what is going on (memory allocations, kernel
# launches, etc.).
option(KITRT_ENABLE_DEBUG "Enable debug mode features for the runtime." OFF)
option(KITRT_ENABLE_VERBOSE "Enable verbose execution mode for debugging." OFF)

option(KITCUDA_ENABLE_NVTX "Enable NVTX profiling within the runtime" OFF)

# Common headers and source files.
set(KITRT_HDRS
  kitrt.h
  debug.h
  memory.h
  memory_map.h)

set(KITRT_SRCS
  kitrt.cpp
  debug.cpp
  memory.cpp
  memory_map.cpp)

set(KITRT kitrt)

add_library(${KITRT} SHARED ${KITRT_SRCS})

if (KITRT_ENABLE_DEBUG)
  target_compile_definitions(${KITRT} -D_KITRT_DEBUG_)
endif()

if (KITRT_ENABLE_VERBOSE)
  target_compile_definitions(${KITRT} -D_KITRT_VERBOSE_)
endif()

target_include_directories(${KITRT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(${KITRT}
  BEFORE PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
  ${LLVM_INCLUDE_DIR}
  ${KITSUNE_SOURCE_DIR}/include
  ${KITSUNE_RUNTIME_INCLUDE_DIR})

set_property(TARGET ${KITRT} APPEND PROPERTY
  INSTALL_RPATH "$ORIGIN/../lib${LLVM_LIBDIR_SUFFIX}")

set_property(TARGET ${KITRT} APPEND PROPERTY
  BUILD_RPATH "$ORIGIN/../lib${LLVM_LIBDIR_SUFFIX}:${PROJECT_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}")

find_library(LIB_LLVM
  NAMES LLVM
  LLVM-${LLVM_VERSION_MAJOR}
  LLVM-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}
  REQUIRED
  PATHS ${LLVM_LIBRARY_DIR}
  NO_DEFAULT_PATH)

# FIXME: We probably don't need to explicitly link pthread, rt, and dl.
# libLLVM is guaranteed to be built because we have explicitly set
# LLVM_BUILD_LLVM_DYLIB to ON.
target_link_libraries(${KITRT}
  PUBLIC
  ${LIB_LLVM}
  pthread
  rt
  dl)

if (KITSUNE_CUDA_ENABLE)
  # FIXME: Instead of looking for CudaToolkit again here, the necessary
  # variables should be passed when this is configured. This ensures that there
  # is only a single call to find_package(CudaToolkit)
  # message(STATUS "Kitsune runtime adding CUDA components to build...")
  # find_package(CUDAToolkit ${KITSUNE_CUDA_VERSION_MIN} REQUIRED)

  find_library(LIB_NVPTX_STATIC nvptxcompiler_static REQUIRED
    PATHS ${CUDAToolkit_LIBRARY_DIR}
    NO_DEFAULT_PATH)

  find_library(LIB_CUDART cudart REQUIRED
    PATHS ${CUDAToolkit_LIBRARY_DIR}
    NO_DEFAULT_PATH)

  find_library(LIB_CUDA cuda REQUIRED
    PATHS ${CUDAToolkit_LIBRARY_DIR}/stubs
    NO_DEFAULT_PATH)

  list(APPEND KITRT_HDRS
    cuda/kitcuda.h
    cuda/kitcuda_dylib.h)

  target_sources(${KITRT} PUBLIC
    cuda/kitcuda.cpp
    cuda/dylib_support.cpp
    cuda/launching.cpp
    cuda/memory.cpp
    cuda/streams.cpp)

  target_compile_definitions(${KITRT} PUBLIC KITRT_CUDA_ENABLED)
  target_include_directories(${KITRT} SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_include_directories(${KITRT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)

  target_link_directories(${KITRT} PUBLIC
    ${CUDAToolkit_LIBRARY_DIR}
    ${CUDAToolkit_LIBRARY_DIR}/stubs)

  target_link_libraries(${KITRT} PUBLIC
    ${LIB_NVPTX_STATIC}
    ${LIB_CUDART}
    ${LIB_CUDA})

  set_property(TARGET ${KITRT} APPEND PROPERTY
    INSTALL_RPATH ${CUDAToolkit_LIBRARY_DIR})

  set_property(TARGET ${KITRT} APPEND PROPERTY
    BUILD_RPATH ${CUDAToolkit_LIBRARY_DIR})

  if (KITCUDA_ENABLE_NVTX)
    target_compile_definitions(${KITRT} PUBLIC KITCUDA_ENABLE_NVTX)

    find_library(LIB_NVTOOLS_EXT nvToolsExt REQUIRED
      PATHS ${CUDAToolkit_LIBRARY_DIR}
      NO_DEFAULT_PATH)

    target_link_libraries(${KITRT} PUBLIC ${LIB_NVTOOLS_EXT})
  endif()
endif()

# FIXME: hip::host depends on clangrt_builtin, but that is not caught by the
# build system currently and the library which causes a build failure if the
# library is not built. Not sure how this is to be fixed, but adding the note
# here as a reminder that it needs to be done.
if (KITSUNE_HIP_ENABLE)
  message(STATUS "Kitsune runtime adding HIP components to build...")
  # FIXME: It may be necessary to hardcode these paths to rocm, but there ought
  # to be better ways of doing this.
  find_package(hip REQUIRED
    PATHS $ENV{ROCM_PATH}/hip $ENV{ROCM_PATH} /opt/rocm/hip /opt/rocm)

  find_library(LIB_HIP_HOST hip::host REQUIRED
    PATHS ${hip_LIBRARY_DIR}
    NO_DEFAULT_PATH)

  list(APPEND KITRT_HDRS
    hip/kithip.h
    hip/kithip_dylib.h)

  target_sources(${KITRT} PUBLIC
    hip/kithip.cpp
    hip/dylib_support.cpp
    hip/launching.cpp
    hip/memory.cpp
    hip/streams.cpp)

  target_compile_definitions(${KITRT} PUBLIC KITRT_HIP_ENABLED)
  target_link_directories(${KITRT} PUBLIC ${hip_LIBRARY_DIR})
  target_link_libraries(${KITRT} PUBLIC hip::host)

  set_property(TARGET ${KITRT} APPEND PROPERTY
    INSTALL_RPATH ${hip_LIBRARY_DIR})
  set_property(TARGET ${KITRT} APPEND PROPERTY
    BUILD_RPATH ${hip_LIBRARY_DIR})

  add_subdirectory(hip)
endif()

# FIXME: Do we need to add the Cuda and Hip directories to the rpath as well?
list(APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

install(TARGETS kitrt
  LIBRARY DESTINATION lib COMPONENT kitrt
  ARCHIVE DESTINATION lib COMPONENT kitrt)

# KITSUNE FIXME: Do we really need to install the runtime headers? Do we intend
# this to be used by third-parties?
install(FILES ${KITRT_HDRS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kitrt)

# KITSUNE FIXME: Do we really need to install the headers for the targets?
install(DIRECTORY cuda hip realm
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kitrt/
  FILES_MATCHING PATTERN "*.h")
