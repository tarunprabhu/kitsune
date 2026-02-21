include(GetClangResourceDir)

add_subdirectory(Config)
add_subdirectory(Core)

get_clang_resource_dir(CLANG_RESOURCE_INCDIR
  PREFIX ${LLVM_LIBRARY_OUTPUT_INTDIR}/..
  SUBDIR include
)

# Add a target to copy kitsune.h into clang's resource directory within the
# build directory. This is needed to run Kitsune from the build directory, most
# crucially to run the core tests.
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
