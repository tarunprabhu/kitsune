// Check that the default lowering of a simple loop is as expected.
//
// RUN: %kitcc --tapir=serial -O1 -S -emit-llvm -o - %s %sysroot \
// RUN:     | FileCheck %s
//
// CHECK-LABEL: @f
// CHECK-SAME: ptr {{[^%]*}}%[[A:[^,]+]],
// CHECK-SAME: i64 {{[^%]*}}%[[N:[^,]+]])
// CHECK: [[ENTRY:.+]]:
// CHECK: [[HEADER:.+]]:
// CHECK: %[[I:.+]] = phi i64
// CHECK: %[[IDX:.+]] = getelementptr {{.*}}i64, ptr %[[A]], i64 %[[I]]
// CHECK: store i64 %[[I]], ptr %[[IDX]]
// CHECK: %[[INC:.+]] = add {{.+}} %[[I]], 1
// CHECK: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[N]]
// CHECK: br i1 %[[CMP]], label %{{[^,]+}}, label %[[HEADER]], !llvm.loop

#include <kitsune.h>

void f(size_t *a, size_t n) {
  // clang-format off
  forall (size_t i = 0; i < n; ++i) {
    a[i] = i;
  }
  // clang-format on
}
