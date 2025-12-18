// -----------------------------------------------------------------------------
// Check that providing a custom config directory without a target-specific
// configuration file in it is ok.
//
// RUN: %kitxx -### --kokkos --tapir=nolo --config-system-dir=%S \
// RUN:     %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CUSTOM-NOEXIST
// RUN: %kitxx -### --kokkos-no-init --tapir=nolo --config-system-dir=%S \
// RUN:     %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CUSTOM-NOEXIST
//
// RUN: %kitxx -### --kokkos --tapir=nolo --config-user-dir=%S \
// RUN:     %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CUSTOM-NOEXIST
// RUN: %kitxx -### --kokkos-no-init --tapir=nolo --config-user-dir=%S \
// RUN:     %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CUSTOM-NOEXIST
//
// CUSTOM-NOEXIST-NOT: Configuration file: {{.*}}/kokkos.cfg
//
// -----------------------------------------------------------------------------
// Check that providing a custom config directory with a target-specific
// configuration file leads to the file being found and the contents used.
//
// RUN: %kitxx -### --kokkos --tapir=nolo \
// RUN:     --config-system-dir=%S/input/cfg1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CUSTOM
// RUN: %kitxx -### --kokkos-no-init --tapir=nolo \
// RUN:     --config-system-dir=%S/input/cfg1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CUSTOM
//
// RUN: %kitxx -### --kokkos --tapir=nolo \
// RUN:     --config-user-dir=%S/input/cfg1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CUSTOM
// RUN: %kitxx -### --kokkos-no-init --tapir=nolo \
// RUN:     --config-user-dir=%S/input/cfg1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CUSTOM
//
// CUSTOM: Configuration file: {{.*}}/kokkos.cfg
// CUSTOM: "-cc1"
// CUSTOM-SAME: "-D" "some_preprocessor_flag"
// CUSTOM-SAME: "-Wsome_compiler_flag"
// CUSTOM-NEXT: "-some_linker_flag"
//
// -----------------------------------------------------------------------------
// If configuration files for both the kokkos and the tapir target are present,
// check that the contents of both are used.
//
// Lit's configuration sets CLANG_NO_DEFAULT_CONFIG=1 before running these
// tests. With this set, the configuration files with the default names will not
// be read, even if a directory in which to look for configuration files has
// been explicitly provided. In most cases, this is exactly what we want, but
// here, we are explicitly checking that the default files are read, so this
// environment variable must be unset.
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN: %kitxx -### --kokkos --tapir=serial -O1 %s 2>&1 \
// RUN:     --config-system-dir=%S/input/cfg2 \
// RUN:     | FileCheck %s -check-prefix=BOTH
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN: %kitxx -### --kokkos-no-init --tapir=serial -O1 %s 2>&1 \
// RUN:     --config-system-dir=%S/input/cfg2 \
// RUN:     | FileCheck %s -check-prefix=BOTH
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN: %kitxx -### --kokkos --tapir=serial -O1 %s 2>&1 \
// RUN:     --config-user-dir=%S/input/cfg2 \
// RUN:     | FileCheck %s -check-prefix=BOTH
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN: %kitxx -### --kokkos-no-init --tapir=serial -O1 %s 2>&1 \
// RUN:     --config-user-dir=%S/input/cfg2 \
// RUN:     | FileCheck %s -check-prefix=BOTH
//
// BOTH: Configuration file: {{.*}}/input/cfg2/kokkos.cfg
// BOTH: Configuration file: {{.*}}/input/cfg2/serial.cfg
// BOTH: "-cc1"
// BOTH-SAME: "-D" "kokkos_preprocessor_flag"
// BOTH-SAME: "-D" "serial_preprocessor_flag"
// BOTH-SAME: "-Wkokkos_compiler_flag"
// BOTH-SAME: "-Wserial_compiler_flag"
// BOTH-NEXT: "-kokkos_linker_flag"
// BOTH-SAME: "-serial_linker_flag"
//
// -----------------------------------------------------------------------------
// If configuration files for the driver, kokkos and the tapir target are all
// present, check that the contents of all of them are used.
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN: %kitxx -### --kokkos --tapir=serial -O1 %s 2>&1 \
// RUN:     --config-system-dir=%S/input/cfg3 \
// RUN:     | FileCheck %s -check-prefix=ALL
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN: %kitxx -### --kokkos-no-init --tapir=serial -O1 %s 2>&1 \
// RUN:     --config-system-dir=%S/input/cfg3 \
// RUN:     | FileCheck %s -check-prefix=ALL
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN: %kitxx -### --kokkos --tapir=serial -O1 %s 2>&1 \
// RUN:     --config-user-dir=%S/input/cfg3 \
// RUN:     | FileCheck %s -check-prefix=ALL
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN: %kitxx -### --kokkos-no-init --tapir=serial -O1 %s 2>&1 \
// RUN:     --config-user-dir=%S/input/cfg3 \
// RUN:     | FileCheck %s -check-prefix=ALL
//
// ALL: Configuration file: {{.*}}/input/cfg3/kit++.cfg
// ALL: Configuration file: {{.*}}/input/cfg3/kokkos.cfg
// ALL: Configuration file: {{.*}}/input/cfg3/serial.cfg
// ALL: "-cc1"
// ALL-SAME: "-D" "driver_preprocessor_flag"
// ALL-SAME: "-D" "kokkos_preprocessor_flag"
// ALL-SAME: "-D" "serial_preprocessor_flag"
// ALL-SAME: "-Wdriver_compiler_flag"
// ALL-SAME: "-Wkokkos_compiler_flag"
// ALL-SAME: "-Wserial_compiler_flag"
// ALL-NEXT: "-driver_linker_flag"
// ALL-SAME: "-kokkos_linker_flag"
// ALL-SAME: "-serial_linker_flag"
