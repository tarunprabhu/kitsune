// RUN: %kitxx -### -ftapir=hip %s 2>&1 | FileCheck %s

// CHECK: "-cc1"
// CHECK-SAME: "-D_tapir_hip_target"

// CHECK: /{{[^ ]*}}ld"
// CHECK-SAME: -lamdhip64
// CHECK-SAME: -lhip::host
// CHECK-SAME: -lkitrt
