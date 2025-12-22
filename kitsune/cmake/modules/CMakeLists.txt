include(FindPrefixFromConfig)
include(GNUInstallPackageDir)

set(KITSUNE_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_PACKAGEDIR}/kitsune"
  CACHE STRING
  "Path for CMake subdirectory for Kitsune")
set(CMAKE_BUILD_PACKAGEDIR
  "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/kitsune")

# ------------------------------------------------------------------------------
# Generate KitsuneConfig.cmake for the build tree.

# We also need to set some LLVM_* variables because they are exposed by
# KitsuneConfig.cmake
set(LLVM_CONFIG_INCLUDE_DIRS
  "${LLVM_MAIN_SRC_DIR}/include"
  "${LLVM_BINARY_DIR}/include")

set(LLVM_CONFIG_LIBRARY_DIRS
  "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}")

set(LLVM_CONFIG_CMAKE_DIR
  "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm")

set(KITSUNE_CONFIG_INCLUDE_DIRS
  "${KITSUNE_SOURCE_DIR}/include"
  "${KITSUNE_BINARY_DIR}/include"
)
list(REMOVE_DUPLICATES KITSUNE_CONFIG_INCLUDE_DIRS)

set(KITSUNE_CONFIG_LIBRARY_DIRS
  "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}"
)
list(REMOVE_DUPLICATES KITSUNE_CONFIG_LIBRARY_DIRS)

set(KITSUNE_CONFIG_BINARY_DIR "${KITSUNE_BINARY_DIR}")
set(KITSUNE_CONFIG_CMAKE_DIR "${CMAKE_BUILD_PACKAGEDIR}")

configure_file(
  KitsuneConfig.cmake.in
  ${CMAKE_BUILD_PACKAGEDIR}/KitsuneConfig.cmake
  @ONLY)

configure_file(
  KitsuneConfigVersion.cmake.in
  ${CMAKE_BUILD_PACKAGEDIR}KitsuneConfigVersion.cmake
  @ONLY)

set(LLVM_CONFIG_INCLUDE_DIRS)
set(LLVM_CONFIG_LIBRARY_DIRS)
set(KITSUNE_CONFIG_CMAKE_DIR)
set(KITSUNE_CONFIG_INCLUDE_DIRS)
set(KITSUNE_CONFIG_LIBRARY_DIRS)

# ------------------------------------------------------------------------------
# Generate KitsuneConfig.cmake for the install tree.

# We also need to set some LLVM_* variables because they are exposed by
# KitsuneConfig.cmake
find_prefix_from_config(KITSUNE_CONFIG_CODE
  KITSUNE_INSTALL_PREFIX "${KITSUNE_INSTALL_PACKAGE_DIR}")

extend_path(LLVM_CONFIG_INCLUDE_DIRS
  "\${KITSUNE_INSTALL_PREFIX}" "${KITSUNE_INSTALL_INCLUDEDIR}")
list(REMOVE_DUPLICATES LLVM_CONFIG_INCLUDE_DIRS)

extend_path(LLVM_CONFIG_LIBRARY_DIRS
  "\${KITSUNE_INSTALL_PREFIX}" "lib\${LLVM_LIBDIR_SUFFIX}")
list(REMOVE_DUPLICATES LLVM_CONFIG_LIBRARY_DIRS)

extend_path(LLVM_CONFIG_CMAKE_DIR
  "\${KITSUNE_INSTALL_PREFIX}" "${LLVM_INSTALL_PACKAGE_DIR}")
list(REMOVE_DUPLICATES LLVM_CONFIG_CMAKE_DIR)

extend_path(KITSUNE_CONFIG_INCLUDE_DIRS
  "\${KITSUNE_INSTALL_PREFIX}" "${CMAKE_INSTALL_INCLUDEDIR}")
list(REMOVE_DUPLICATES KITSUNE_CONFIG_INCLUDE_DIRS)

extend_path(KITSUNE_CONFIG_LIBRARY_DIRS
  "\${KITSUNE_INSTALL_PREFIX}" "lib\${LLVM_LIBDIR_SUFFIX}")
list(REMOVE_DUPLICATES KITSUNE_CONFIG_LIBRARY_DIRS)

extend_path(KITSUNE_CONFIG_CMAKE_DIR
  "\${KITSUNE_INSTALL_PREFIX}" "${KITSUNE_INSTALL_PACKAGE_DIR}")
list(REMOVE_DUPLICATES KITSUNE_CONFIG_CMAKE_DIR)

configure_file(
  KitsuneConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/KitsuneConfig.cmake
  @ONLY)

configure_file(
  KitsuneConfigVersion.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/KitsuneConfigVersion.cmake
  @ONLY)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/KitsuneConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/KitsuneConfigVersion.cmake
    DESTINATION ${KITSUNE_INSTALL_PACKAGE_DIR})
endif()

set(LLVM_CONFIG_INCLUDE_DIRS)
set(LLVM_CONFIG_LIBRARY_DIRS)
set(KITSUNE_CONFIG_CODE)
set(KITSUNE_CONFIG_CMAKE_DIR)
set(KITSUNE_CONFIG_INCLUDE_DIRS)
set(KITSUNE_CONFIG_LIBRARY_DIRS)
