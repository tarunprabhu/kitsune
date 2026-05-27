; REQUIRES: kitsune-qthreads
;
; Check that intrinsics that map to Kitsune's qthreads runtime are lowered
; correctly. If more intrinsics are created, they should be added here to test
; basic intrinsic lowering.
;
; RUN: opt --tapir=qthreads -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

@gbuf = external global [7 x float]

; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: call void @__kitqthr_initialize()
; CHECK-NEXT: call void @__kitqthr_launch(ptr @f, i64 0, i64 128, i64 1, ptr @gbuf)
; CHECK-NEXT: call i64 @__kitqthr_reduce_num_partials(i64 %[[N]])
; CHECK-NEXT: call void @__kitqthr_finalize()

define void @f(ptr %buf, i64 %n) {
  call void @llvm.kit.runtime.initialize(i32 32)
  call void @llvm.kit.cpu.threads.launch(i32 32, ptr @f, i64 0, i64 128, i64 1, ptr @gbuf)
  %1 = call i64 @llvm.kit.reduce.num.partials(i32 32, i64 %n)
  call void @llvm.kit.runtime.finalize(i32 32)
  ret void
}
