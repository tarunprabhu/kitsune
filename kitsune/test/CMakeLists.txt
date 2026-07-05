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
  KITSUNE_QTHREADS_ENABLED
  KITSUNE_REALM_ENABLED
  KITSUNE_BUILD_EXAMPLES
)

if (KITSUNE_BUILD_EXAMPLES)
  set(SHLIB_EXT "${CMAKE_SHARED_LIBRARY_SUFFIX}")

  get_target_property(KITSUNE_EMB_PASS_PLUGIN_NAME kit-emb-pass-plugin-demo
    OUTPUT_NAME)
  get_target_property(KITSUNE_EMB_PASS_PLUGIN_DIR kit-emb-pass-plugin-demo
    LIBRARY_OUTPUT_DIRECTORY)
  set(KITSUNE_EMB_PASS_PLUGIN_DEMO_DYLIB
    "${KITSUNE_EMB_PASS_PLUGIN_DIR}/${KITSUNE_EMB_PASS_PLUGIN_NAME}${SHLIB_EXT}")

  get_target_property(KITSUNE_PASS_PLUGIN_NAME kit-pass-plugin-demo
    OUTPUT_NAME)
  get_target_property(KITSUNE_PASS_PLUGIN_DIR kit-pass-plugin-demo
    LIBRARY_OUTPUT_DIRECTORY)
  set(KITSUNE_PASS_PLUGIN_DEMO_DYLIB
    "${KITSUNE_PASS_PLUGIN_DIR}/${KITSUNE_PASS_PLUGIN_NAME}${SHLIB_EXT}")

  get_target_property(KITSUNE_TT_PLUGIN_NAME kit-tt-plugin-demo
    OUTPUT_NAME)
  get_target_property(KITSUNE_TT_PLUGIN_DIR kit-tt-plugin-demo
    LIBRARY_OUTPUT_DIRECTORY)
  set(KITSUNE_TT_PLUGIN_DEMO_DYLIB
    "${KITSUNE_TT_PLUGIN_DIR}/${KITSUNE_TT_PLUGIN_NAME}${SHLIB_EXT}")
endif ()

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
  "KITSUNE_EXAMPLES_SOURCE_DIR"
  "KITSUNE_EXAMPLES_BINARY_DIR"
  "KITSUNE_EMB_PASS_PLUGIN_DEMO_DYLIB"
  "KITSUNE_PASS_PLUGIN_DEMO_DYLIB"
  "KITSUNE_TT_PLUGIN_DEMO_DYLIB"
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
  ${KITRT}
  KitsuneUnitTests
  kit-config
  kit-enc
  kit-mbc
  kit-sort
  kitsune-installed-headers
  clang
  clang-offload-bundler
  count
  FileCheck
  llc
  lld
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
  list(APPEND KITSUNE_TEST_DEPS flang-rt)
  list(APPEND KITSUNE_TEST_DEPS ${KITSUNE_Fortran_FRONTEND})
endif()

if (KITSUNE_BUILD_EXAMPLES)
  list(APPEND KITSUNE_TEST_DEPS kit-emb-pass-plugin-demo)
  list(APPEND KITSUNE_TEST_DEPS kit-pass-plugin-demo)
  list(APPEND KITSUNE_TEST_DEPS kit-tt-plugin-demo)
endif ()

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

# The build system automatically creates targets of the form
# `check-<dir1>-<dir2>-...-<dirN>` where `<dir2>` is a subdirectory of `<dir1>`,
# `<dir3>` is a subdirectory of `<dir2>` and so on. These can be used to run
# the tests in a specific subdirectory. However, typing the full path can get
# tedious. For convenience, create some custom targets that save a bit of
# typing.

# Targets of the form `check-kitsune-<TT>` where <TT> is the name of a tapir
# target. These convenience targets will only be created for enabled tapir
# targets.
foreach(TT IN LISTS KITSUNE_ENABLED_TAPIR_TARGETS)
  add_custom_target(check-kitsune-${TT} DEPENDS check-kitsune-tapir-${TT})
endforeach()
add_custom_target(check-kitsune-nolo DEPENDS check-kitsune-tapir-nolo)

# Targets of the form `check-kitsune-<LANG>` where <LANG> is the name of a
# language for which a frontend exists. The table below lists some the currently
# known languages and the corresponding target generated by the code below.
#
#     LANGUAGE | TARGET
#     --------------------------------
#     C        | check-kitsune-c
#     C++      | check-kitsune-cxx
#     Fortran  | check-kitsune-fortran
#     --------------------------------
#
# These convenience targets will only be created for enabled languages.
#
foreach(LANG IN LISTS KITSUNE_ENABLED_LANGS)
  add_custom_target(check-kitsune-${LANG} DEPENDS check-kitsune-frontend-${LANG})
endforeach ()
if (KITSUNE_KOKKOS_ENABLED)
  add_custom_target(check-kitsune-kokkos DEPENDS check-kitsune-frontend-kokkos)
endif ()

add_custom_target(check-kitsune-pass-plugin DEPENDS check-kitsune-plugins-pass)
add_custom_target(check-kitsune-tt-plugin DEPENDS check-kitsune-plugins-tt)
add_custom_target(check-kitsune-llc DEPENDS check-kitsune-tools-llc)
add_custom_target(check-kitsune-lld DEPENDS check-kitsune-tools-lld)
add_custom_target(check-kitsune-opt DEPENDS check-kitsune-tools-opt)
add_custom_target(check-kit-config DEPENDS check-kitsune-tools-kit-config)
add_custom_target(check-kit-enc DEPENDS check-kitsune-tools-kit-enc)
add_custom_target(check-kit-mbc DEPENDS check-kitsune-tools-kit-mbc)
add_custom_target(check-kit-sort DEPENDS check-kitsune-tools-kit-sort)
