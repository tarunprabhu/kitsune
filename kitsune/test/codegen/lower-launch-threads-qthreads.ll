; REQUIRES: kitsune-qthreads
;
; Check that the launch of threads by the qthreads tapir target is lowered
; correctly.
;
; RUN: opt --tapir=qthreads -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @launch
; CHECK: call void @__kitqthr_launch(ptr nonnull @f, i64 0, i64 %n, i64 11, ptr %args)

target triple = "x86_64-pc-linux-gnu"

define internal void @f(i64 %start, i64 %end, ptr %args) {
  ret void
}

define void @launch(i64 %n, ptr %args) {
  call void @llvm.kit.cpu.threads.launch(i32 32, ptr nonnull @f, i64 0, i64 %n, i64 11, ptr %args)
  ret void
}
