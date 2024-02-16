// Check that the default target-specific configuration file is always found.
// RUN: %kitxx -### -ftapir=openmp %s 2>&1 | FileCheck %s -check-prefix=CHECK-DEFAULT-CONFIG

// Check that providing a custom config directory without a target-specific
// configuration file is ok.
// RUN: %kitxx -### -ftapir=openmp --config-kitsune-dir=%S/../ %s 2>&1 | FileCheck %s -check-prefix=CHECK-CUSTOM-NOEXIST

// Check that providing a custom config directory with a target-specific
// configuration file leads to the file being found and the contents used.
// RUN: %kitxx -### -ftapir=openmp --config-kitsune-dir=%S %s 2>&1 | FileCheck %s -check-prefix=CHECK-CUSTOM

// CHECK-DEFAULT-CONFIG: Configuration file: {{.*}}/openmp.cfg
// CHECK-CUSTOM-NOEXIST-NOT: Configuration file: {{.*}}/openmp.cfg
// CHECK-CUSTOM: Configuration file: {{.*}}/openmp/openmp.cfg
// CHECK-CUSTOM: "-cc1"
// CHECK-CUSTOM-SAME: "-D" "some_preprocessor_flag"
// CHECK-CUSTOM-SAME: "-Wsome_compiler_flag"
// CHECK-CUSTOM: /{{[^ ]*}}ld
// CHECK-CUSTOM-SAME: "-some_linker_flag"
