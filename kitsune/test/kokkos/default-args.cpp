// RUN: %kitxx -### -ftapir=kokkos %s 2>&1 | FileCheck %s

// CHECK: "-cc1"
// CHECK-SAME: -I{{[^ ]*}}/include/kokkos

// CHECK: /{{[^ ]*}}ld
// CHECK-SAME: -lkokkos
// CHECK-SAME: -lkokkoscore
// CHECK-SAME: -lkitrt
