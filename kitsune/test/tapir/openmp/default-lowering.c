// Check that the default lowering of a simple loop is as expected. The loop
// body must be outlined. The outlined function is passed to Kitsune's
// launch threads intrinsic. We check the output immediately after loop-spawning
// because optimizations run after loop-spawning will remove unused arguments
// and we are more interested in testing exactly what the openmp tapir target
// generates.
//
// RUN: %kitcc --tapir=openmp -O1 %s %sysroot -S -emit-llvm -o /dev/null \
// RUN:     -mllvm -print-after=loop-spawning 2>&1 \
// RUN:     | FileCheck %s
//
// CHECK-LABEL: @f(
// CHECK-SAME: i64 {{[^%]*}}%[[N:[^)]+]])
// CHECK-NEXT: [[ENTRY:.+]]:
// CHECK-NEXT: %[[ARGS:.+]] = alloca { i64, i64 }
// CHECK: [[BODY:.+]]:
// CHECK: %[[THRDS32:.+]] = tail call i32 @llvm.kit.cpu.num.threads(i32 512)
// CHECK: %[[NUM_THREADS:.+]] = sext i32 %[[THRDS32]] to i64
// CHECK: %[[PER_THREAD:.+]] = udiv i64 {{.+}}, %[[NUM_THREADS]]
// CHECK: %[[ARGPOS0:.+]] = getelementptr {{.*}}, ptr %[[ARGS]], i32 0, i32 0
// CHECK: store i64 %[[PER_THREAD]], ptr %[[ARGPOS0]]
// CHECK: %[[ARGPOS1:.+]] = getelementptr {{.*}}, ptr %[[ARGS]], i32 0, i32 1
// CHECK: store i64 %[[N]], ptr %[[ARGPOS1]]
// CHECK: call void @llvm.kit.cpu.threads.launch(
// CHECK-SAME: i32 512,
// CHECK-SAME: ptr @[[OUTLINED:[^,]+]],
// CHECK-SAME: i64 0,
// CHECK-SAME: i64 %[[NUM_THREADS]],
// CHECK-SAME: ptr %[[ARGS]])
//
// CHECK: define internal fastcc void @[[OUTLINED:[A-Za-z0-9._-]+]](
// CHECK-SAME: i64 %[[START:[^,]+]],
// CHECK-SAME: i64 %[[END:[^,]+]],
// CHECK-SAME: ptr {{[^%]*}}%[[ARGS:[^)]+]])
// CHECK: %[[OUTER:.+]] = phi i64
// CHECK: %[[I:.+]] = phi i64
// CHECK: call void @ext({{.*}} %[[I]])
// CHECK: add {{.*}}i64 %[[I]], 1
// CHECK: br {{.+}}, label %{{.+}}, label %{{.+}}

#include <kitsune.h>

void ext(size_t);

void f(size_t n) {
  // clang-format off
  forall (size_t i = 0; i < n; ++i) {
    ext(i);
  }
  // clang-format on
}
