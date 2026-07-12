# This target is only present so the system tests can be added as dependencies
# to it. This target itself is a dependency for the `check-kitsune` and
# `check-kitrt` lit targets.
add_custom_target(KitrtTests)

# Add a "system" test to the list of kitrt's system tests. This will generate an
# executable, linked against libkitrt, from the single source file, SOURCE. This
# executable will be invoked from the RUN directives present in SOURCE. The name
# of the resulting executable will be ${SOURCE}.test.
function(add_kitrt_systest source)
  cmake_parse_arguments(ARG "" "" "INCDIRS;LIBS" ${ARGN})

  set(name ${source}.test)
  add_executable(${name} ${source})

  target_include_directories(${name} PUBLIC ${KITRT_SOURCE_DIR})
  if (ARG_INCDIRS)
    target_include_directories(${name} PUBLIC ${ARG_INCDIRS})
  endif ()

  target_link_libraries(${name} PUBLIC ${KITRT_DSO})
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  if (ARG_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LIBS})
  endif ()

  add_dependencies(KitrtTests ${name})
endfunction()

add_subdirectory(common)
add_subdirectory(openmp)
add_subdirectory(pthreads)

if (KITSUNE_CUDA_ENABLED)
  add_subdirectory(cuda)
endif ()

if (KITSUNE_HIP_ENABLED)
  add_subdirectory(hip)
endif ()

if (KITSUNE_OPENCILK_ENABLED)
  add_subdirectory(opencilk)
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
