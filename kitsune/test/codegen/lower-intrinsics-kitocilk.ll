; REQUIRES: kitsune-opencilk
;
; Check that intrinsics that map to Kitsune's opencilk runtime are lowered
; correctly. If more intrinsics are created, they should be added here to test
; basic intrinsic lowering.
;
; RUN: opt --tapir=opencilk -passes='kit-lower-intrinsics' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[BUF:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: call void @__kitocilk_initialize()
; CHECK-NEXT: call i64 @__kitocilk_reduce_num_partials(i64 %[[N]])
; CHECK-NEXT: call void @__kitocilk_finalize()

; This needs a triple in order to correctly initialize the target library.
target triple = "x86_64-pc-linux-gnu"

@gbuf = external global [7 x float]

define void @f(ptr %buf, i64 %n) {
  call void @llvm.kit.initialize(i32 8)
  %1 = call i64 @llvm.kit.reduce.num.partials(i32 8, i64 %n)
  call void @llvm.kit.finalize(i32 8)
  ret void
}
