; Check that intrinsics that map to Kitsune's openmp runtime are lowered
; correctly. If more intrinsics are created, they should be added here to test
; basic intrinsic lowering.
;
; RUN: opt --tapir=openmp -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: call void @__kitomp_initialize()
; CHECK-NEXT: call void @__kitrt_enable_verbose_mode()
; CHECK-NEXT: call void @__kitomp_launch(ptr @f, i64 0, i64 128, i64 1, ptr @gbuf)
; CHECK-NEXT: call i64 @__kitomp_reduce_num_partials(i64 %[[N]])
; CHECK-NEXT: call void @__kitomp_finalize()

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

@gbuf = external global [7 x float]

define void @f(ptr %buf, i64 %n) {
  call void @llvm.kit.runtime.initialize(i32 512)
  call void @llvm.kit.runtime.set.verbose(i32 2, i8 1)
  call void @llvm.kit.runtime.set.verbose(i32 2, i8 0)
  call void @llvm.kit.cpu.threads.launch(i32 512, ptr @f, i64 0, i64 128, i64 1, ptr @gbuf)
  %1 = call i64 @llvm.kit.reduce.num.partials(i32 512, i64 %n)
  call void @llvm.kit.runtime.finalize(i32 512)
  ret void
}
