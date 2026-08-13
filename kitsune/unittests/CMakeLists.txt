# All unit tests are added as dependencies of this target. This target is itself
# a dependency of the `check-kitsune` lit targets.
add_custom_target(KitsuneUnitTests)

# Add a unittest to the suite of Kitsune's unit tests. The code below shows how
# this function may be used.
#
#     set(LLVM_LINK_COMPONENTS L1 L2)
#     set(CLANG_LINK_COMPONENTS C1 C2)
#
#     add_kitsune_unittest(ExecName
#       Source1.cpp
#       Source2.cpp
#       ...
#
#       <other named arguments>
#     )
#
# This will create an executable named ExecNameTests using the source files
# Source1.cpp, Source2.cpp, and so on. Additional named arguments, indicated
# by '<other named arguments>', are passed as is to `add_unittest` (defined in
# `AddLLVM.cmake`). ExecNameTests will be linked against gtest.
#
# Additional libraries to be linked into the executable can be provided by
# setting the `LLVM_LINK_COMPONENTS` and `CLANG_LINK_COMPONENTS` variables.
# In the example above, this will link libLLVML1, libLLVML2, libclangC1, and
# libclangC2 into ExecNameTests.
function(add_kitsune_unittest basename)
  # We suffix "Tests" to the base name because GoogleTest has been configured to
  # expect all unit test executables to end with this string. See
  # ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py.
  set(name ${basename}Tests)

  # EXPORT_SYMBOLS here is required, otherwise, spurious failures can occur
  # if LLVM_LINK_LLVM_DYLIB=ON
  add_unittest(KitsuneUnitTests ${name} ${ARGN} EXPORT_SYMBOLS)

  # We have to explicitly do this for clang. add_unittests will do the
  # equivalent for the entries in ${LINK_LLVM_COMPONENTS}.
  list(TRANSFORM CLANG_LINK_COMPONENTS PREPEND "clang")
  target_link_libraries(${name} PRIVATE ${CLANG_LINK_COMPONENTS})
  target_link_libraries(${name} PRIVATE LLVMTestingSupport)
endfunction()

# gtest macros like EXPECT_TRUE are expanded to a single line multi-statement
# code with if/else. E.g.
#
#   if (...)
#     EXPECT_TRUE(...)
#
# will be expanded into something like:
#
#   if(...)
#     switch (0) case 0: default: if (...) ; else return;;
#
# GCC may emit false positive dangling-else warnings for such code. However,
# such warnings are actually against LLVM's style guide. Disable the warning for
# GCC so that one can enable -Werror.
if (CMAKE_COMPILER_IS_GNUCXX)
  list(APPEND LLVM_COMPILE_FLAGS "-Wno-dangling-else")
endif ()

add_subdirectory(Analysis)
add_subdirectory(Clang)
add_subdirectory(Config)
add_subdirectory(Core)
add_subdirectory(Passes)
add_subdirectory(Shared)
add_subdirectory(Support)

# We probably don't need to do this after the unit tests are configured, but we
# might as well do it late, just in case.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  "KITSUNE_BINARY_DIR"
  "KITSUNE_SOURCE_DIR"
)

add_lit_testsuite(check-kitsune-unit "Running the Kitsune unit tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS KitsuneUnitTests)
