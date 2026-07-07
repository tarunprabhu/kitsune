# This target is only present so the system tests can be added as dependencies
# to it. This target itself is a dependency for the main `check-kitsune` target
# that will run all Kitsune tests. It is also a dependency for the `check-kitrt`
# target that will only run the runtime tests.
add_custom_target(KitrtTests)
set_target_properties(KitrtTests PROPERTIES FOLDER "Kitrt/Tests")

# Add a "system" test to the list of kitrt's system tests. These are
# executables, linked against libkitrt, that will be invoked by kitrt's lit
# tests. The single source file is compiled to form the executable. The name of
# the executable is the same as the source file with the ".test" suffix
# appended to it.
function(add_kitrt_systest source)
  set(test_name ${source}.test)
  add_executable(${test_name} ${source})

  target_include_directories(${test_name} PUBLIC ${KITSUNE_SOURCE_DIR}/runtime/)
  target_link_libraries(${test_name} PUBLIC ${KITRT})
  set_target_properties(${test_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_dependencies(KitrtTests ${test_name})
endfunction()

add_subdirectory(openmp)
