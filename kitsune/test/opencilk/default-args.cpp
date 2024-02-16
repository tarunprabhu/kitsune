// RUN: %kitxx -### -ftapir=opencilk %s 2>&1 | FileCheck %s

// CHECK: /{{[^ ]*}}ld
// CHECK-SAME: -lopencilk
// CHECK-SAME: -lkitrt
