include (CheckLinkerFlag)

check_linker_flag(CXX "LINKER:-z,now" LINKER_SUPPORTS_Z_NOW)

# This target is only present so the system tests can be added as dependencies
# to it. This target itself is a dependency for the `check-kitsune` and
# `check-kitrt` lit targets.
add_custom_target(KitrtTests)

# Add a "system" test to the list of kitrt's system tests. This will generate an
# executable, linked against libkitrt, from the single source file, SOURCE. This
# executable will be invoked from the RUN directives present in SOURCE.
function(add_kitrt_systest source)
  cmake_parse_arguments(ARG "" "" "INCDIRS;LIBS" ${ARGN})

  # CMake requires a globally unique target name. In several cases, there may be
  # multiple test files with the same name. This is desirable - for instance,
  # when we are testing the behavior of the __kit*_num_threads function exposed
  # by different tapir-target-specific runtimes. Since these executables are
  # only intended for internal use, we generate a name that contains the name
  # of the tapir target in addition to a specific suffix.
  get_filename_component(dir "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
  set(target ${dir}-${source}.test)

  add_executable(${target} ${source})
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  # FIXME: Reorganize the runtime code so we have a KITRT_TEST_DIR
  target_include_directories(${target} PUBLIC
    ${KITRT_SOURCE_DIR}
    ${KITRT_SOURCE_DIR}/test/include)
  if (ARG_INCDIRS)
    target_include_directories(${target} PUBLIC ${ARG_INCDIRS} SYSTEM)
  endif ()

  target_link_libraries(${target} PUBLIC ${KITRT_DSO})
  if (ARG_LIBS)
    target_link_libraries(${target} PUBLIC ${ARG_LIBS})
  endif ()

  if (LINKER_SUPPORTS_Z_NOW)
    target_link_options(${target} PRIVATE "LINKER:-z,now")
  endif ()

  set_target_properties(${target} PROPERTIES
    C_STANDARD 23
    CXX_STANDARD 17)

  add_dependencies(KitrtTests ${target})
endfunction()

# Convenience function to add a number of system tests. TESTS is a list of
# source files - with one tests created for each source file. Any additional
# named arguments passed to this are forwarded as is to `add_kitrt_systest`.
function(add_kitrt_systests tests)
  foreach (test IN LISTS tests)
    add_kitrt_systest(${test} ${ARGN})
  endforeach()
endfunction ()

add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(openmp)
add_subdirectory(pthreads)
add_subdirectory(timer)

if (KITSUNE_CUDA_ENABLED)
  add_subdirectory(cuda)
endif ()

if (KITSUNE_HIP_ENABLED)
  add_subdirectory(hip)
endif ()

if (KITSUNE_OPENCILK_ENABLED)
  add_subdirectory(opencilk)
endif ()

if (KITSUNE_PAPI_ENABLED)
  add_subdirectory(papi)
endif ()

if (KITSUNE_QTHREADS_ENABLED)
  add_subdirectory(qthreads)
endif ()

# We probably don't need to do this after the unit tests are configured, but we
# might as well do it late, just in case.
llvm_canonicalize_cmake_booleans(
  KITSUNE_KOKKOS_ENABLED
  KITSUNE_CUDA_ENABLED
  KITSUNE_HIP_ENABLED
  KITSUNE_LAMBDA_ENABLED
  KITSUNE_OMPTASK_ENABLED
  KITSUNE_OPENCILK_ENABLED
  KITSUNE_QTHREADS_ENABLED
  KITSUNE_REALM_ENABLED
  KITSUNE_PAPI_ENABLED
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  "LLVM_TOOLS_DIR"
  "KITRT_BINARY_DIR"
  "KITRT_SOURCE_DIR"
)

# We want the check-kitrt target to run both Kitrt's lit and unit tests.
add_lit_testsuite(check-kitrt "Running the Kitrt regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ${KITRT_BINARY_DIR}/unittests
  DEPENDS KitrtTests KitrtUnitTests)

# The first argument to `add_lit_testsuites` is the project name. In the rest
# of LLVM, this is always passed in all-uppercase. It is not clear why, but we
# follow the same convention. This creates `check-kitrt-*` targets for each
# subdirectory of ${CMAKE_CURRENT_SOURCE_DIR}
add_lit_testsuites(KITRT ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS KitrtTests)
