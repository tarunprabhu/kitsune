add_custom_target(KitsuneUnitTests)
set_target_properties(KitsuneUnitTests PROPERTIES FOLDER "Kitsune/Tests")

function(add_kitsune_unittest test_dirname)
  add_unittest(KitsuneUnitTests ${test_dirname} ${ARGN})
endfunction()

function(add_kitsune_unittest_with_input_files test_dirname)
  add_unittest_with_input_files(KitsuneUnitTests ${test_dirname} ${ARGN})
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
add_subdirectory(Config)
add_subdirectory(Core)
add_subdirectory(Frontend)
add_subdirectory(Passes)
add_subdirectory(Support)
