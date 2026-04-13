; Check that the launch of threads by the openmp tapir target is lowered
; correctly.
;
; RUN: opt --tapir=openmp -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @launch
; CHECK: call void @__kitomp_launch(ptr nonnull @f, i64 0, i64 %n, i64 11, ptr %args)

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

define internal void @f(i64 %start, i64 %end, ptr %args) {
  ret void
}

define void @launch(i64 %n, ptr %args) {
  call void @llvm.kit.launch.threads(i32 512, ptr nonnull @f, i64 0, i64 %n, i64 11, ptr %args)
  ret void
}
