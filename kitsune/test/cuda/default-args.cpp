// RUN: %kitxx -### -ftapir=cuda %s 2>&1 | FileCheck %s

// CHECK: "-cc1"
// CHECK-SAME: "-D_tapir_cuda_target"

// CHECK: /{{[^ ]*}}ld
// CHECK-SAME: -lcudart
// CHECK-SAME: -lcuda
// CHECK-SAME: -lkitrt
