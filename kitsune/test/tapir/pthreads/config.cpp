// -----------------------------------------------------------------------------
// Check that the default target-specific configuration file is always found.
//
// RUN: %kitxx -### --tapir=pthreads -O1 %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-DEFAULT-CONFIG
//
// CHECK-DEFAULT-CONFIG: Configuration file: {{.*}}/pthreads.cfg
//
// -----------------------------------------------------------------------------
// Check that providing a custom config directory without a target-specific
// configuration file is ok.
//
// RUN: %kitxx -### --tapir=pthreads -O1 --config-kitsune-dir=%S/ %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-CUSTOM-NOEXIST
//
// COM: %kitxx -### --tapir=pthreads -O1 --config-user-dir=%S/ %s 2>&1 \
// COM:     | FileCheck %s -check-prefix=CHECK-CUSTOM-NOEXIST
//
// CHECK-CUSTOM-NOEXIST-NOT: Configuration file: {{.*}}/pthreads.cfg
//
// -----------------------------------------------------------------------------
// Check that providing a custom config directory with a target-specific
// configuration file leads to the file being found and the contents used.
//
// RUN: %kitxx -### --tapir=pthreads -O1 --config-kitsune-dir=%S/input %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-CUSTOM
//
// RUN: %kitxx -### --tapir=pthreads -O1 --config-user-dir=%S/input %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-CUSTOM
//
// CHECK-CUSTOM: Configuration file: {{.*}}/input/pthreads.cfg
// CHECK-CUSTOM: "-cc1"
// CHECK-CUSTOM-SAME: "-D" "some_preprocessor_flag"
// CHECK-CUSTOM-SAME: "-Wsome_compiler_flag"
// CHECK-CUSTOM-NEXT: "-some_linker_flag"
