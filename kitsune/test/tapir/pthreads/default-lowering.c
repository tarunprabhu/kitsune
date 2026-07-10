// Check that the default lowering of a simple loop is as expected. The loop
// body must be outlined. The outlined function is passed to Kitsune's
// async launch threads intrinsic. The threads are synchronized immediately.
//
// RUN: %kitcc --tapir=pthreads -O1 %s %sysroot \
// RUN:     -S -emit-llvm -o - \
// RUN:     | FileCheck %s
//
// CHECK-LABEL: @f
// CHECK-SAME: i64 {{.*}}%[[N:[^)]+]])
// CHECK: [[ENTRY:.+]]:
// CHECK: %[[ARGS:.+]] = alloca { i64, i64 }
// CHECK: [[BODY:.+]]:
// CHECK: %[[THRDS32:.+]] = tail call i32 @llvm.kit.cpu.num.threads(i32 1024)
// CHECK: %[[NUM_THREADS:.+]] = sext i32 %[[THRDS32]] to i64
// CHECK: %[[PER_THREAD:.+]] = udiv i64 {{.+}}, %[[NUM_THREADS]]
// CHECK: store i64 %[[PER_THREAD]], ptr %[[ARGS]]
// CHECK: %[[ARGPOS1:.+]] = getelementptr {{.*}}, ptr %[[ARGS]], i64 8
// CHECK: store i64 %[[N]], ptr %[[ARGPOS1]]
// CHECK: %[[CTX:.+]] = call ptr @llvm.kit.async.cpu.threads.launch(
// CHECK-SAME: i32 1024,
// CHECK-SAME: ptr @[[OUTLINED:[^,]+]],
// CHECK-SAME: i64 0,
// CHECK-SAME: i64 %[[NUM_THREADS]],
// CHECK-SAME: ptr %[[ARGS]])
// CHECK: call void @llvm.kit.cpu.threads.sync(i32 1024, ptr %[[CTX]])
// CHECK: br label %[[END:.+]]
// CHECK: [[END]]:
// CHECK: ret void
//
// CHECK: define internal fastcc void @[[OUTLINED]](
// CHECK-SAME: i64 %[[START:[^,]+]],
// CHECK-SAME: i64 %[[END:[^,]+]],
// CHECK-SAME: ptr {{.*}}%[[ARGS:[^)]+]])
// CHECK: %[[OUTER:.+]] = phi i64
// CHECK: %[[I:.+]] = phi i64
// CHECK: call void @ext({{.*}} %[[I]])
// CHECK: add {{.*}}i64 %[[I]], 1
// CHECK: br {{.+}}, label %{{.+}}, label {{.+}}

#include <kitsune.h>

void ext(size_t);

void f(size_t n) {
  // clang-format off
  forall (size_t i = 0; i < n; ++i) {
    ext(i);
  }
  // clang-format on
}
