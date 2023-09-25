# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
#

set(KITRT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KITRT_INCLUDE_DIR ${KITSUNE_RUNTIME_DIR})

# There is support for two different levels of debugging.  The general
# debug mode (KITRT_ENABLE_DEBUG) will enable runtime assertion checks
# that will cause the runtime to both check and abort on any error
# conditions from the underlying runtimes (e.g., CUDA, HIP, etc.).  The
# second option (KITRT_ENABLE_VERBOSE) will report runtime details and
# give a general feel for what is going on (memory allocations, kernel
# launches, etc.).
option(KITRT_ENABLE_DEBUG "Enable debug mode features for the runtime." OFF)
option(KITRT_ENABLE_VERBOSE "Enable verbose execution mode for debuggin." OFF)
set(KITSUNE_BITCODE_INSTALL_DIR  "lib/bitcode/" CACHE STRING
  "Location to install Kitsune's runtime bitcode files.")


include_directories(${KITSUNE_SOURCE_DIR}/include)
include_directories(${KITSUNE_RUNTIME_INCLUDE_DIR})

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
  memory_map.cpp
  support.cpp)

set(KITRT_LINK_LIBS "")
add_library(kitrt SHARED ${KITRT_SRCS})
set(CMAKE_LINK_WHAT_YOU_USE ON)

if (KITRT_ENABLE_DEBUG)
  target_compile_definitions(kitrt _KITRT_DEBUG_)
endif()

if (KITRT_ENABLE_VERBOSE)
  target_compile_definitions(kitrt _KITRT_VERBOSE_)
endif()

target_include_directories(kitrt
  BEFORE PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
  ${LLVM_INCLUDE_DIR}
  ${KITSUNE_SOURCE_DIR}/include
  ${KITSUNE_RUNTIME_INCLUDE_DIR})

target_link_libraries(kitrt
  PUBLIC
  # FIXME: Make sure this links to the libLLVM.so that was built here.
  LLVM)

if (KITRT_ENABLE_CUDA)
  message(STATUS "Kitsune runtime adding CUDA components to build...")

  find_package(CUDAToolkit REQUIRED)
  find_library(NVPTXCOMPILER_STATIC nvptxcompiler_static REQUIRED
    PATHS ${CUDAToolkit_LIBRARY_DIR}
    NO_DEFAULT_PATH)

  list(APPEND KITRT_HDRS
    cuda/cuda.h
    cuda/llvm-cuda.h)

  target_sources(kitrt
    PUBLIC cuda/cuda.cpp cuda/llvm-cuda.cpp)

  target_compile_definitions(kitrt
    PUBLIC KITRT_CUDA_ENABLED)

  target_include_directories(kitrt
    PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

  target_link_libraries(kitrt
    PUBLIC ${NVPTXCOMPILER_STATIC} CUDA::cudart CUDA::cuda_driver)
endif()

if (KITRT_ENABLE_HIP)
  message(STATUS "Kitsune runtime adding HIP components to build...")

  # TODO: This probably needs external settings in order to
  # manage modules and other ways to install rocm...
  list(APPEND CMAKE_PREFIX_PATH
    PATHS $ENV{ROCM_PATH} $ENV{ROCM_PATH} /opt/rocm/hip /opt/rocm
  )
  find_package(hip CONFIG REQUIRED)

  list(APPEND KITRT_HDRS
    hip/hip.h
    hip/llvm-hip.h)

  target_sources(kitrt
    PUBLIC hip/hip.cpp hip/llvm-hip.cpp)

  target_compile_definitions(kitrt
    PUBLIC KITRT_HIP_ENABLED)

  target_link_libraries(kitrt
    PUBLIC hip::host)

  add_subdirectory(hip)
endif()

target_link_libraries(kitrt
  PUBLIC dl rt pthread)

set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

# FIXME: LLVM now requires C++17. Is this necessary any longer? We will probably
# not support older versions of LLVM anyway.
set_property(TARGET kitrt PROPERTY CXX_STANDARD 17)

install(TARGETS kitrt
  LIBRARY DESTINATION lib COMPONENT kitrt
  ARCHIVE DESTINATION lib COMPONENT kitrt)
install(FILES ${KITRT_HDRS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kitrt)
# FIXME: Are there any headers here that are not added to KITRT_HDRS? If so,
# why not? These also currently get copied unconditionally. Should they be
# guarded depending on the runtimes that are actually built?
install(DIRECTORY cuda hip realm
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kitrt/
  FILES_MATCHING PATTERN "*.h")

