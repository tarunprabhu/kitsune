// Kitsune always uses LLD that was built alongside Kitsune. Overriding this
// linker is not allowed.
//
// RUN: %kitxx -### --tapir=serial -O2 %s 2>&1 | FileCheck %s
//
// CHECK: /ld{{(64)?}}.lld
// CHECK-SAME: -dynamic-linker
