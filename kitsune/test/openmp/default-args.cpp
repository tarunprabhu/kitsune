// RUN: %kitxx -### -ftapir=openmp %s 2>&1 | FileCheck %s

// CHECK: /{{[^ ]*}}ld
// CHECK-SAME: -lomp
// CHECK-SAME: -lkitrt
