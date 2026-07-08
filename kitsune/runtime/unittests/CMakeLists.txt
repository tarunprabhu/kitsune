# All unit tests are added as dependencies of this target. This target is itself
# a dependency of the `check-kitrt` and `check-kitsune` lit targets.
add_custom_target(KitrtUnitTests)
set_target_properties(KitrtUnitTests PROPERTIES FOLDER "Kitrt/UnitTests")

# Add a unittest to the suite of kitrt's unit tests. This will create an
# executable named NAME that will be linked against gtest and kitrt. The code
# below shows how this function may be used.
#
#     add_kitsune_unittest(ExecName
#       Source1.cpp
#       Source2.cpp
#       ...
#
#       <other named arguments>
#     )
#
# This will create an executable named ExecName using the source files
# Source1.cpp, Source2.cpp, and so on. Additional named arguments, indicated
# by '<other named arguments>', are passed as is to `add_unittest` (defined in
# `AddLLVM.cmake`).
function(add_kitrt_unittest name)
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
