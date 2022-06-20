#
# Copyright (c) 2022 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune+tapir project.  It is released under
# the LLVM license.
#

# The sources will be passed as additional arguments to this function. This
# should not be called directly outside of this file.
function(add_plugin_sources PROP)
  get_property(srcs GLOBAL PROPERTY ${PROP})
  foreach(src IN LISTS ARGN)
    list(APPEND srcs "${src}")
  endforeach()
  set_property(GLOBAL PROPERTY ${PROP} ${srcs})
endfunction(add_plugin_sources PROP)

# This should be called to add sources to be compiled into a clang plugin that
# will operate on the source.
function(add_kitsune_plugin_frontend_sources)
  add_plugin_sources(kitsune_plugin_frontend_srcs ${ARGN})
endfunction(add_kitsune_plugin_frontend_sources)

# This should be called to add sources to be compiled into a LLVM pass plugin.
function(add_kitsune_plugin_backend_sources)
  add_plugin_sources(kitsune_plugin_backend_srcs ${ARGN})
endfunction(add_kitsune_plugin_backend_sources)

function(add_kitsune_plugin_lib SONAME SOURCES LIBS)
  add_library(${SONAME} SHARED ${SOURCES})

  target_link_libraries(${SONAME} PUBLIC ${LIBS})

  # Don't give the plugin a lib prefix.
  set_target_properties(${SONAME} PROPERTIES PREFIX "")

  install(TARGETS ${SONAME} DESTINATION lib/kitsune)
endfunction(add_kitsune_plugin_lib)

# Link with the shared objects in the build directory during build. This should
# make it possible to run tests. Once installed, switch to using the installed
# library. Make the rpath explicit so it doesn't pick up any system library.
set(CMAKE_BUILD_RPATH ${LLVM_LIB_DIR})
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

# This seems to be necessary to prevent undefined variable linker errors.
add_definitions(-DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING)

# This is lib in the build directory.
set(LLVM_LIB_DIR ${LLVM_BINARY_DIR}/lib)
set(PLUGIN_LIBS
  ${LLVM_LIB_DIR}/libclang.so
  ${LLVM_LIB_DIR}/libclang-cpp.so
  ${LLVM_LIB_DIR}/libLLVM-${LLVM_VERSION_MAJOR}.so
)

add_compile_options(-fno-exceptions)
include_directories(BEFORE
  ${PROJECT_SOURCE_DIR}/../clang/include
  ${PROJECT_SOURCE_DIR}/../llvm/include
  ${LLVM_BINARY_DIR}/tools/clang/include
  ${LLVM_BINARY_DIR}/include)

# Contains all the source files that will be used in the frontend plugin.
set_property(GLOBAL PROPERTY kitsune_plugin_frontend_srcs PluginRegistry.cpp)

# Contains all the source files that will be used in the backend (LLVM pass)
# plugin.
set_property(GLOBAL PROPERTY kitsune_plugin_backend_srcs "")

# Add subdirectories for all the plugins first so the source files list can
# be populated. This must be done before the shared library targets are
# declared.
add_subdirectory(ConstGlobalVars)

get_property(FRONTEND_SOURCES GLOBAL PROPERTY kitsune_plugin_frontend_srcs)
get_property(BACKEND_SOURCES GLOBAL PROPERTY kitsune_plugin_backend_srcs)

add_kitsune_plugin_lib("KitsunePlugin" "${FRONTEND_SOURCES}" "${PLUGIN_LIBS}")
add_kitsune_plugin_lib("KitsunePassPlugin" "${BACKEND_SOURCES}" "${PLUGIN_LIBS}")
