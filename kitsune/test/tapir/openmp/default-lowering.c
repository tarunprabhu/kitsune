// Check that the default lowering of a simple loop is as expected. The loop
// body must be outlined. The outlined function is passed to Kitsune's
// launch threads intrinsic. We check the output immediately after loop-spawning
// because optimizations run after loop-spawning will remove unused arguments
// and we are more interested in testing exactly what the openmp tapir target
// generates.
//
// RUN: %kitcc --tapir=openmp -O1 %s %sysroot -S -emit-llvm -o /dev/null \
// RUN:     -mllvm -print-after=loop-spawning 2>&1 \
// RUN:     | FileCheck --dump-input=fail %s
//
// CHECK-LABEL: @f(
// CHECK-SAME: ptr {{[^%]*}}%[[A:[^,]+]],
// CHECK-SAME: i64 {{[^%]*}}%[[N:[^)]+]])
// CHECK-NEXT: [[ENTRY:.+]]:
// CHECK-NEXT: %[[ARGS:.+]] = alloca { ptr }
// CHECK: [[BODY:.+]]:
// CHECK: %[[GS:[0-9]+]] = {{.*}}call i64 @llvm.tapir.loop.grainsize
// CHECK: %[[ARGPOS:.+]] = getelementptr {{.*}}, ptr %[[ARGS]]
// CHECK: store ptr %[[A]], ptr %[[ARGPOS]]
// CHECK: call void @llvm.kit.launch.threads(
// CHECK-SAME: i32 512,
// CHECK-SAME: ptr @[[OUTLINED:[^,]+]],
// CHECK-SAME: i64 0,
// CHECK-SAME: i64 %[[N]],
// CHECK-SAME: i64 %[[GS]],
// CHECK-SAME: ptr %[[ARGS]])
//
// CHECK: define internal fastcc void @[[OUTLINED:[A-Za-z0-9._-]+]](
// CHECK-SAME: i64 %[[START:[^,]+]],
// CHECK-SAME: i64 %[[END:[^,]+]],
// CHECK-SAME: i64 %[[GRAINSIZE:[^,]+]],
// CHECK-SAME: ptr {{[^%]*}}%[[ARGS:[^)]+]])
// CHECK: [[ENTRY:.+]]:
// CHECK: br label %[[HEADER:.+]]
// CHECK: [[HEADER]]:
// CHECK: %[[I:.+]] = phi i64
// CHECK: br label %[[BODY:.+]]
// CHECK: [[BODY]]:
// CHECK: %[[APOS:.+]] = getelementptr {{.*}}i64, ptr {{.+}}, i64 %[[I]]
// CHECK: store i64 %[[I]], ptr %[[APOS]]
// CHECK: add {{.*}}i64 %[[I]], 1
// CHECK: br {{.+}}, label %[[END:.+]], label %[[HEADER]]
// CHECK: [[END]]:
// CHECK: ret void

#include <kitsune.h>

void f(size_t *a, size_t n) {
  // clang-format off
  forall (size_t i = 0; i < n; ++i) {
    a[i] = i;
  }
  // clang-format on
}
