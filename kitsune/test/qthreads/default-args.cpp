// RUN: %kitxx -### -ftapir=qthreads %s 2>&1 | FileCheck %s

// CHECK: /{{[^ ]*}}ld
// CHECK-SAME: -lqthreads
// CHECK-SAME: -lkitrt
