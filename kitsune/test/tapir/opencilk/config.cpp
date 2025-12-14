// -----------------------------------------------------------------------------
// Check that the default target-specific configuration file is always found.
//
// RUN: %kitxx -### --tapir=opencilk -O1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=DEFAULT
//
// DEFAULT: Configuration file: {{.*}}/opencilk.cfg
//
// -----------------------------------------------------------------------------
// Check that providing a custom config directory without a target-specific
// configuration file is ok.
//
// RUN: %kitxx -### --tapir=opencilk -O1 --config-kitsune-dir=%S %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CUSTOM-NOEXIST
//
// COM: %kitxx -### --tapir=opencilk -O1 --config-user-dir=%S %s 2>&1 \
// COM:     | FileCheck %s -check-prefix=CUSTOM-NOEXIST
//
// CUSTOM-NOEXIST-NOT: Configuration file: {{.*}}/opencilk.cfg
//
// -----------------------------------------------------------------------------
// Check that providing a custom config directory with a target-specific
// configuration file leads to the file being found and the contents used.
//
// RUN: %kitxx -### --tapir=opencilk -O1 --config-kitsune-dir=%S/input %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CUSTOM
//
// RUN: %kitxx -### --tapir=opencilk -O1 --config-user-dir=%S/input %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CUSTOM
//
// CUSTOM: Configuration file: {{.*}}/input/opencilk.cfg
// CUSTOM: "-cc1"
// CUSTOM-SAME: "-D" "some_preprocessor_flag"
// CUSTOM-SAME: "-Wsome_compiler_flag"
// CUSTOM-NEXT: "-some_linker_flag"
//
// -----------------------------------------------------------------------------
// If configuration files for both the driver and the tapir target are present,
// check that the contents of both are used and the default options are
// preserved.
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN: %kitxx -### --tapir=opencilk -O1 %s 2>&1 \
// RUN:     --config-kitsune-dir=%S/input/cfgs \
// RUN:     | FileCheck %s -check-prefix=BOTH
//
// RUN: env CLANG_NO_DEFAULT_CONFIG= \
// RUN: %kitxx -### --tapir=opencilk -O1 %s 2>&1 \
// RUN:     --config-user-dir=%S/input/cfgs \
// RUN:     | FileCheck %s -check-prefix=BOTH
//
// BOTH: Configuration file: {{.*}}/input/cfgs/kit++.cfg
// BOTH: Configuration file: {{.*}}/input/cfgs/opencilk.cfg
// BOTH: "-cc1"
// BOTH-SAME: "-D" "driver_preprocessor_flag"
// BOTH-SAME: "-D" "tapir_preprocessor_flag"
// BOTH-SAME: "-Wdriver_compiler_flag"
// BOTH-SAME: "-Wtapir_compiler_flag"
// BOTH-NEXT: "-driver_linker_flag"
// BOTH-SAME: "-tapir_linker_flag"
