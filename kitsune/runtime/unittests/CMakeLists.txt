# All unit tests are added as dependencies of this target. This target is itself
# a dependency of the `check-kitrt` and `check-kitsune` lit targets.
add_custom_target(KitrtUnitTests)

# Add a unittest to the suite of kitrt's unit tests. This will create an
# executable named ${basename}Tests that will be linked against gtest and kitrt.
# The code below shows how this function may be used.
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
# `AddLLVM.cmake`). ExecNameTests will be linked against gtest and kitrt.
function(add_kitrt_unittest basename)
  # We suffix "Tests" to the base name because GoogleTest has been configured to
  # expect all unit test executables to end with this string. See
  # ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py.
  set(name ${basename}Tests)

  # EXPORT_SYMBOLS here is required, otherwise, spurious failures can occur
  # if LLVM_LINK_LLVM_DYLIB=ON
  add_unittest(KitrtUnitTests ${name} ${ARGN} EXPORT_SYMBOLS)

  target_include_directories(${name} PUBLIC ${KITRT_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE LLVMTestingSupport)
  target_link_libraries(${name} PUBLIC ${KITRT_DSO})
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

add_subdirectory(common)

# We probably don't need to do this after the unit tests are configured, but we
# might as well do it late, just in case.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  "KITRT_BINARY_DIR"
  "KITRT_SOURCE_DIR"
)

add_lit_testsuite(check-kitrt-unit "Running the Kitrt unit tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS KitrtUnitTests)
