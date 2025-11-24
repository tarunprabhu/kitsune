// Check that the default lowering of a simple loop is as expected.
//
// RUN: %kitcc --tapir=serial -O1 -S -emit-llvm -o - %s %sysroot \
// RUN:     | FileCheck %s
//
// CHECK: define {{.*}}void @f(ptr {{.*}}%[[A:.+]], i32 {{.*}}%[[SCALE:.+]], i64 {{.*}}%[[N:.+]])
// CHECK: [[ENTRY:.+]]:
// CHECK: br {{.+}}, label %[[END:.+]], label %[[BODY:.+]]
// CHECK: [[BODY]]:
// CHECK: %[[I:.+]] = phi i64
// CHECK: %[[IDX:.+]] = getelementptr {{.*}}i32, ptr %[[A]], i64 %[[I]]
// CHECK: %[[V:.+]] = load i32, ptr %[[IDX]]
// CHECK: %[[SCALED:.+]] = mul {{.*}}%[[V]], %[[SCALE]]
// CHECK: store i32 %[[SCALED]], ptr %[[IDX]]
// CHECK: %[[INC:.+]] = add {{.+}} %[[I]], 1
// CHECK: %[[CMP:.+]] = icmp
// CHECK: br i1 %[[CMP]], label %[[END]], label %[[BODY]], !llvm.loop
// CHECK: [[END:.+]]:
// CHECK: ret void

#include <kitsune.h>

void f(int *a, int scale, size_t n) {
  // clang-format off
  forall (size_t i = 0; i < n; ++i) {
    a[i] *= scale;
  }
  // clang-format on
}
