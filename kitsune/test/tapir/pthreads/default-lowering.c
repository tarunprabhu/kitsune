// Check that the default lowering of a simple loop is as expected. The loop
// body must be outlined. The outlined function is passed to Kitsune's
// async launch threads intrinsic. The threads are synchronized immediately.
//
// RUN: %kitcc --tapir=pthreads -O1 %s %sysroot \
// RUN:     -S -emit-llvm -o - \
// RUN:     | FileCheck %s
//
// CHECK-LABEL: @f
// CHECK-SAME: ptr {{.*}}%[[A:[^,]+]],
// CHECK-SAME: i64 {{.*}}%[[N:[^)]+]])
// CHECK: [[ENTRY:.+]]:
// CHECK: %[[ARGS:.+]] = alloca { ptr }
// CHECK: [[BODY:.+]]:
// CHECK: store ptr %[[A]], ptr %[[ARGS]]
// CHECK: %[[CTX:.+]] = call ptr @llvm.kit.async.launch.threads(
// CHECK-SAME: i32 1024,
// CHECK-SAME: ptr @[[OUTLINED:[^,]+]],
// CHECK-SAME: i64 0,
// CHECK-SAME: i64 %[[N]],
// CHECK-SAME: i64 0,
// CHECK-SAME: ptr %[[ARGS]])
// CHECK: call void @llvm.kit.sync.threads(i32 1024, ptr %[[CTX]])
// CHECK: br label %[[END:.+]]
// CHECK: [[END]]:
// CHECK: ret void
//
// CHECK: define internal fastcc void @[[OUTLINED]](
// CHECK-SAME: i64 %[[START:[^,]+]],
// CHECK-SAME: i64 %[[END:[^,]+]],
// CHECK-SAME: i64 %[[GRAINSIZE:[^,]+]],
// CHECK-SAME: ptr {{.*}}%[[ARGS:[^)]+]])
// CHECK: [[ENTRY:.+]]:
// CHECK: [[HEADER:.+]]:
// CHECK: %[[I:.+]] = phi i64
// CHECK: %[[IDX:.+]] = getelementptr {{.*}}i64, ptr {{.+}}, i64 %[[I]]
// CHECK: store i64 %[[I]], ptr %[[IDX]]
// CHECK: add {{.*}}i64 %[[I]], 1
// CHECK: br {{.+}}, label %[[END:.+]], label %[[HEADER]]

#include <kitsune.h>

void f(size_t *a, size_t n) {
  // clang-format off
  forall (size_t i = 0; i < n; ++i) {
    a[i] = i;
  }
  // clang-format on
}
