# This target is only present so the system tests can be added as dependencies
# to it. This target itself is a dependency for the `check-kitsune` and
# `check-kitrt` lit targets.
add_custom_target(KitrtTests)
set_target_properties(KitrtTests PROPERTIES FOLDER "Kitrt/Tests")

# Add a "system" test to the list of kitrt's system tests. These are
# executables, linked against libkitrt, that will be invoked by kitrt's lit
# tests. The single source file is compiled to form the executable. The name of
# the executable is the same as the source file with the ".test" suffix
# appended to it.
function(add_kitrt_systest source)
  set(name ${source}.test)
  add_executable(${name} ${source})

  target_include_directories(${name} PUBLIC ${KITRT_SOURCE_DIR})
  target_link_libraries(${name} PUBLIC ${KITRT_DSO})
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_dependencies(KitrtTests ${name})
endfunction()

add_subdirectory(openmp)
