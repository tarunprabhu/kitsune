include(GetClangResourceDir)

# Create a single target that other targets, especially those in LLVM, can
# depend on to ensure that all tablegen files in Kitsune have been processed.
# This target will depend on the targets created by kitsune_tablegen. Those
# dependencies will be added in the appropriate subdirectories where
# kitsune_tablegen is actually called.
add_custom_target(kitsune_generated)

add_subdirectory(Config)
add_subdirectory(Core)

get_clang_resource_dir(CLANG_RESOURCE_INT_INCDIR
  PREFIX ${CMAKE_BINARY_DIR}
  SUBDIR include)

get_clang_resource_dir(CLANG_RESOURCE_INSTALL_INCDIR
  PREFIX ${CMAKE_INSTALL_PREFIX}
  SUBDIR include)

add_custom_command(
  OUTPUT ${CLANG_RESOURCE_INT_INCDIR}/kitsune.h
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${CMAKE_CURRENT_SOURCE_DIR}/kitsune.h
  ${CLANG_RESOURCE_INT_INCDIR}/kitsune.h
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kitsune.h
  VERBATIM USES_TERMINAL)

# Add a target to copy kitsune.h into clang's resource directory within the
# build directory. This is needed to run Kitsune from the build directory.
add_custom_target(kitsune-installed-headers ALL
  DEPENDS ${CLANG_RESOURCE_INT_INCDIR}/kitsune.h)

# kitsune.h should always be installed since it is needed by applications that
# are compiled with kitsune.
install(FILES ${CLANG_RESOURCE_INT_INCDIR}/kitsune.h
  DESTINATION ${CLANG_RESOURCE_INSTALL_INCDIR})
