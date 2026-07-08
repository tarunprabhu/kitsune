# All unit tests are added as dependencies of this target. This target is itself
# a dependency of the `check-kitsune` lit targets.
add_custom_target(KitsuneUnitTests)
set_target_properties(KitsuneUnitTests PROPERTIES FOLDER "Kitsune/UnitTests")

# Add a unittest to the suite of Kitsune's unit tests. This will create an
# executable named NAME that will be linked against gtest. The code below shows
# how this function may be used.
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
# This will create an executable named ExecName using the source files
# Source1.cpp, Source2.cpp, and so on. Additional named arguments, indicated
# by '<other named arguments>', are passed as is to `add_unittest` (defined in
# `AddLLVM.cmake`).
#
# Additional libraries to be linked into the executable can be provided by
# setting the `LLVM_LINK_COMPONENTS` and `CLANG_LINK_COMPONENTS` variables.
# In the example above, this will link libLLVML1, libLLVML2, libclangC1, and
# libclangC2 into the executable ExecName. Note how a prefix is automatically
# added depending on the variable containing the library.
function(add_kitsune_unittest name)
  add_unittest(KitsuneUnitTests ${name} ${ARGN} EXPORT_SYMBOLS)

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
add_subdirectory(Support)
