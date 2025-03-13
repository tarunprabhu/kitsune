// Check that non-constant globals are handled correctly.
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o - %s 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK-DAG: @[[HOSTVAR:.+]] = external {{.+}} i32
// CHECK-DAG: @[[DEVVAR_NAME:.+]] = private constant [12 x i8] c"v137_devvar\00"
//
// CHECK: define {{.+}} @fill
// CHECK: %[[DEVPTR:.+]] = call i64 @__kitcuda_get_global_symbol(ptr nonnull @_kitsune_fatbin_cuda, ptr nonnull @[[DEVVAR_NAME]])
// CHECK: call void @__kitcuda_memcpy_sym_to_device(ptr nonnull @[[HOSTVAR]], i64 %[[DEVPTR]], i64 4)
// CHECK: call ptr @__kitcuda_launch_kernel
//
// FIXME: Check that the variable is registered correctly.

#include <cstdlib>
#include <kitsune.h>

extern int v137;

extern "C" void fill(int* c, size_t n) {
  forall (size_t i = 0; i < n; ++i) {
    c[i] = v137;
  }
}
