// Check that a launch call and a fat binary are present in the host.
//
// RUN: %kitxx --tapir=cuda --tapir-cuda-arch=sm_72 -S -emit-llvm -O2 -o - %s \
// RUN:     | FileCheck %s
//
// CHECK: @__kitsune_fatbin_cuda = {{.+}} constant [{{[0-9]+}} x i8] c"
// CHECK: define {{.+}} @f
// CHECK: %[[TS:.+]] = call {{.+}} @__kitcuda_launch_kernel(
// CHECK: call {{.+}} @__kitcuda_sync_thread_stream(ptr %[[TS]])
// CHECK: ret void
// CHECK-NEXT: }

#include <kitsune.h>

extern "C" void f(int *c, size_t n) {
  forall(size_t i = 0; i < n; ++i) c[i] = n;
}
