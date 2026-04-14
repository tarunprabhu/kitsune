// -----------------------------------------------------------------------------
// Check that providing a custom config directory without a target-specific
// configuration file is ok.
//
// RUN: %kitxx -### --tapir=pthreads -O1 --config-system-dir=%S/ %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CUSTOM-NOEXIST
//
// COM: %kitxx -### --tapir=pthreads -O1 --config-user-dir=%S/ %s 2>&1 \
// COM:     | FileCheck %s -check-prefix=CUSTOM-NOEXIST
//
// CUSTOM-NOEXIST-NOT: Configuration file: {{.*}}/pthreads.cfg
//
// -----------------------------------------------------------------------------
// Check that providing a custom config directory with a target-specific
// configuration file leads to the file being found and the contents used.
//
// RUN: %kitxx -### --tapir=pthreads -O1 --config-system-dir=%S/input %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CUSTOM
//
// RUN: %kitxx -### --tapir=pthreads -O1 --config-user-dir=%S/input %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CUSTOM
//
// CUSTOM: Configuration file: {{.*}}/input/pthreads.cfg
// CUSTOM: "-cc1"
// CUSTOM-SAME: "-D" "pthreads_preprocessor_flag"
// CUSTOM-SAME: "-Wpthreads_compiler_flag"
// CUSTOM-NEXT: "-pthreads_linker_flag"
//
// -----------------------------------------------------------------------------
// If configuration files for both the driver and the tapir target are present,
// check that the contents of both are used and the default options are
// preserved.
//
// Lit's configuration sets CLANG_NO_DEFAULT_CONFIG=1 before running these
// tests. With this set, the configuration files with the default names will not
// be read, even if a directory in which to look for configuration files has
// been explicitly provided. In most cases, this is exactly what we want, but
// here, we are explicitly checking that the default files are read, so this
// environment variable must be unset.
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN: %kitxx -### --tapir=pthreads -O1 %s 2>&1 \
// RUN:     --config-system-dir=%S/input/cfgs \
// RUN:     | FileCheck %s -check-prefix=BOTH
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN: %kitxx -### --tapir=pthreads -O1 %s 2>&1 \
// RUN:     --config-user-dir=%S/input/cfgs \
// RUN:     | FileCheck %s -check-prefix=BOTH
//
// BOTH: Configuration file: {{.*}}/input/cfgs/kit++.cfg
// BOTH: Configuration file: {{.*}}/input/cfgs/pthreads.cfg
// BOTH: "-cc1"
// BOTH-SAME: "-D" "pthreads_driver_preprocessor_flag"
// BOTH-SAME: "-D" "pthreads_custom_preprocessor_flag"
// BOTH-SAME: "-Wpthreads_driver_compiler_flag"
// BOTH-SAME: "-Wpthreads_custom_compiler_flag"
// BOTH-NEXT: "-pthreads_driver_linker_flag"
// BOTH-SAME: "-pthreads_custom_linker_flag"
