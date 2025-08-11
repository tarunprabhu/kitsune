# Test runner infrastructure for Kitsune. This configures the Kitsune test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

llvm_canonicalize_cmake_booleans(
  KITSUNE_C_ENABLED
  KITSUNE_CXX_ENABLED
  KITSUNE_Fortran_ENABLED
  KITSUNE_KOKKOS_ENABLED
  KITSUNE_CUDA_ENABLED
  KITSUNE_HIP_ENABLED
  KITSUNE_LAMBDA_ENABLED
  KITSUNE_OMPTASK_ENABLED
  KITSUNE_OPENCILK_ENABLED
  KITSUNE_OPENMP_ENABLED
  KITSUNE_QTHREADS_ENABLED
  KITSUNE_REALM_ENABLED
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  "LLVM_SOURCE_DIR"
  "LLVM_BINARY_DIR"
  "LLVM_TOOLS_DIR"
  "LLVM_LIBS_DIR"
  "SHLIBDIR"
  "LLVM_LIT_TOOLS_DIR"
  "LLVM_EXTERNAL_LIT"
  "KITSUNE_BINARY_DIR"
  "KITSUNE_SOURCE_DIR"
  "CURRENT_TOOLS_DIR"
  "CMAKE_LIBRARY_OUTPUT_DIRECTORY"
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/unit/lit.cfg.py
  PATHS
  "LLVM_SOURCE_DIR"
  "LLVM_BINARY_DIR"
  "LLVM_TOOLS_DIR"
  "SHLIBDIR"
  "KITSUNE_BINARY_DIR"
  "KITSUNE_SOURCE_DIR"
)

set(KITSUNE_TEST_PARAMS
  USE_Z3_SOLVER=0
)

list(APPEND KITSUNE_TEST_DEPS
  ${KITSUNE_LIBNAME}
  KitsuneUnitTests
  kit-config
  kit-enc
  kit-mbc
  kit-mdc
  kitsune-resource-headers
  clang
  count
  FileCheck
  llc
  lld
  llvm-ar
  llvm-as
  llvm-config
  llvm-dis
  not
  opt
)
if (KITSUNE_C_ENABLED)
  list(APPEND KITSUNE_TEST_DEPS ${KITSUNE_C_FRONTEND})
endif()
if (KITSUNE_CXX_ENABLED)
  list(APPEND KITSUNE_TEST_DEPS ${KITSUNE_CXX_FRONTEND})
endif()
if (KITSUNE_Fortran_ENABLED)
  list(APPEND KITSUNE_TEST_DEPS flang)
  list(APPEND KITSUNE_TEST_DEPS ${KITSUNE_Fortran_FRONTEND})
endif()
if (KITSUNE_KOKKOS_ENABLED)
  list(APPEND KITSUNE_TEST_DEPS kokkos-copy-to-build-dir)
endif()
if (KITSUNE_OPENCILK_ENABLED)
  list(APPEND KITSUNE_TEST_DEPS cheetah-copy-to-build-dir)
endif()
if (KITSUNE_QTHREADS_ENABLED)
  list(APPEND KITSUNE_TEST_DEPS qthreads-copy-to-build-dir)
endif()
if (KITSUNE_REALM_ENABLED)
  list(APPEND KITSUNE_TEST_DEPS realm-copy-to-build-dir)
endif()

if(TARGET llvm-lto)
  list(APPEND KITSUNE_TEST_DEPS llvm-lto)
endif()

add_custom_target(kitsune-test-depends DEPENDS ${KITSUNE_TEST_DEPS})
set_target_properties(kitsune-test-depends PROPERTIES FOLDER "Kitsune tests")

add_lit_testsuite(check-kitsune "Running the Kitsune regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS ${KITSUNE_TEST_PARAMS}
  DEPENDS ${KITSUNE_TEST_DEPS}
)
set_target_properties(check-kitsune PROPERTIES FOLDER "Kitsune tests")

add_lit_testsuites(KITSUNE ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS ${KITSUNE_TEST_PARAMS}
  DEPENDS ${KITSUNE_TEST_DEPS}
  FOLDER "Kitsune tests"
)

# Add a target that is used to run the tests for the LLVM projects that are
# relevant for kitsune.
add_custom_target(check-kitsune-all DEPENDS
  check-llvm
  check-clang
  check-lld
  check-kitsune)
if (KITSUNE_Fortran_ENABLED)
  add_dependencies(check-kitsune-all
    check-mlir
    check-flang)
endif ()

# These custom targets are purely for convenience. Currently, to run the tests
# for just a single tapir target, we need to type check-kitsune-tapir-<TT>,
# where TT is the name of one of the tapir targets. This is a result of the
# organization of tests in this directory. check-kitsune-<TT> is more ergonomic
# and saves a bit of typing during development.
foreach(TT IN LISTS KITSUNE_ENABLED_TAPIR_TARGETS)
  add_custom_target(check-kitsune-${TT}
    DEPENDS check-kitsune-tapir-${TT})
endforeach()
