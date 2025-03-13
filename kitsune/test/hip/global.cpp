// Check that non-constant globals are handled correctly.
//
// RUN: %kitxx --tapir=hip -O2 -S -emit-llvm -o - %s 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK-DAG: @[[HOSTVAR:.+]] = external {{.+}}global i32
// CHECK-DAG: @[[DEVVAR_NAME:[^ ]+]] = private constant [12 x i8] c"v137_devvar\00"
//
// CHECK: define {{.+}} @fill
// CHECK: %[[DEVPTR:.+]] = call ptr @__kithip_get_global_symbol(ptr nonnull @_kitsune_fatbin_hip, ptr nonnull @[[DEVVAR_NAME]]
// CHECK: call void @__kithip_memcpy_sym_to_device(ptr nonnull @[[HOSTVAR]], ptr %[[DEVPTR]], i64 4)
// CHECK: call ptr @__kithip_launch_kernel
//
// FIXME: Check that the variable is registered with the runtime.

#include <cstdlib>
#include <kitsune.h>

extern int v137;

extern "C" void fill(int* c, size_t n) {
  forall (size_t i = 0; i < n; ++i) {
    c[i] = v137;
  }
}
