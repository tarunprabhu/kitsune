// Check that a launch call and a fat binary are present in the host.
//
// RUN: %kitxx --tapir=hip -S -emit-llvm -O2 -o - %s \
// RUN:     | FileCheck %s
//
// CHECK: @_kitsune_fatbin_hip = {{.+}} constant [{{[0-9]+}} x i8] c"
// CHECK: define {{.+}} @f
// CHECK: %[[TS:.+]] = call {{.+}} @__kithip_launch_kernel(
// CHECK: call {{.+}} @__kithip_sync_thread_stream(ptr %[[TS]])
// CHECK: ret void
// CHECK-NEXT: }

#include <kitsune.h>

extern "C" void f(int *c, size_t n) {
  forall(size_t i = 0; i < n; ++i) c[i] = n;
}
