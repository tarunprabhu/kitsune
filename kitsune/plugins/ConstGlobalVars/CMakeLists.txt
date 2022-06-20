#
# Copyright (c) 2022 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune+tapir project.  It is released under
# the LLVM license.
#

set(FRONTEND_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/ConstGlobalVarsPlugin.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ConstGlobalVarsContext.cpp)

set(BACKEND_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/ConstGlobalVarsPass.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ConstGlobalVarsContext.cpp)

add_kitsune_plugin_frontend_sources(${FRONTEND_SOURCES})
add_kitsune_plugin_backend_sources(${BACKEND_SOURCES})
