#
# Copyright (c) 2022 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune+tapir project.  It is released under
# the LLVM license.
#

add_kitsune_plugin_common_sources(
  ${CMAKE_CURRENT_SOURCE_DIR}/ConstGlobalVarsContext.cpp)

add_kitsune_plugin_frontend_sources(
  ${CMAKE_CURRENT_SOURCE_DIR}/ConstGlobalVarsPlugin.cpp)

add_kitsune_plugin_backend_sources(
  ${CMAKE_CURRENT_SOURCE_DIR}/ConstGlobalVarsPass.cpp)
