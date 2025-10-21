; Check that the launch of threads by the pthreads tapir target is lowered
; correctly.
;
; RUN: opt --tapir=pthreads -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @launch
; CHECK: call ptr @__kitpthr_launch(ptr nonnull @f, i64 0, i64 %n, i64 11, ptr %args)

target triple = "x86_64-unknown-linux-gnu"

define internal void @f(i64 %start, i64 %end, ptr %args) {
  ret void
}

define void @launch(i64 %n, ptr %args) {
  %ctx = call ptr @llvm.kit.async.launch.threads(i32 1024, ptr nonnull @f, i64 0, i64 %n, i64 11, ptr %args)
  ret void
}
