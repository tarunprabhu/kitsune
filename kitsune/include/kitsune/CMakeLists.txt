#
# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the Kitsune project.  It is released under the LLVM
# license.
#

include(GetClangResourceDir)

# Kitsune is registered as a project and will show up as a "tool" alongside
# clang. But we actually want this configuration file in a place where clang
# will pick it up.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Config/config.h.cmake
  ${CMAKE_BINARY_DIR}/include/kitsune/Config/config.h
  @ONLY)

get_clang_resource_dir(CLANG_RESOURCE_INCDIR
  PREFIX ${LLVM_LIBRARY_OUTPUT_INTDIR}/..
  SUBDIR include
)

add_custom_command(OUTPUT ${CLANG_RESOURCE_INCDIR}/kitsune.h
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${CMAKE_CURRENT_SOURCE_DIR}/kitsune.h ${CLANG_RESOURCE_INCDIR}/kitsune.h
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kitsune.h
  VERBATIM USES_TERMINAL)

add_custom_target(kitsune-resource-headers ALL
  DEPENDS ${CLANG_RESOURCE_INCDIR}/kitsune.h)

# kitsune.h should always be installed since it is needed by applications that
# are compiled with kitsune.
install(FILES kitsune.h
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/clang/${LLVM_VERSION_MAJOR}/include
  COMPONENT kitsune-resource-headers)

# However, the headers in include/kitsune are only installed if we are doing a
# "non-toolchain-only" installation.
if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  add_custom_target(kitsune-headers)

  install(DIRECTORY ${CMAKE_BINARY_DIR}/include/kitsune
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT kitsune-headers)
endif ()
