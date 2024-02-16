// RUN: %kitxx -### -ftapir=realm %s 2>&1 | FileCheck %s

// CHECK: /{{[^ ]*}}ld
// CHECK-SAME: -lrealm
// CHECK-SAME: -lrealm-abi
// CHECK-SAME: -lkitrt
