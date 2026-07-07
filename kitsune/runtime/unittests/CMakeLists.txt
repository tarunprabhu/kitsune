# add_custom_target(KitrtUnitTests)
# set_target_properties(KitrtUnitTests PROPERTIES FOLDER "Kitrt/UnitTests")

# Add a unittest with name TEST_NAME to the suite of kitrt's unit tests.
# This will create an executable named TEST_NAME that will be linked against
# gtest. The code below shows how this function may be used.
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
# Source1.cpp, Source2.cpp, and so on. The additional named arguments, indicated
# by '<other named arguments>' in the example above are passed as is to
# `add_unittest` (defined in `AddLLVM.cmake`).
#
# The only library these should depend on is libkitrt. This will be added by
# this function.
function(add_kitrt_unittest test_name)
  add_unittest(KitrtUnitTests ${test_name} ${ARGN} EXPORT_SYMBOLS)

  target_include_directories(${test_name} PUBLIC ${KITSUNE_SOURCE_DIR}/runtime/)
  target_link_libraries(${test_name} PRIVATE LLVMTestingSupport)
  target_link_libraries(${test_name} PUBLIC ${KITRT})
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

add_kitrt_unittest(KitrtUnitTests
  EnvVarUtilsTest.cpp
  MiscUtilsTest.cpp
)
