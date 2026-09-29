; Check that calls to kit.gpu.ipt are lowered correctly when the iterations per
; thread are known at compile-time.
;
; RUN: opt -passes=kit-lower-intrinsics-early -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: %[[INC:.+]] = add i64 17, 1
; CHECK-NEXT: ret i64 %[[INC]]

@0 = internal constant i64 17

define i64 @f() {
   %ipt = call i64 @llvm.kit.gpu.ipt(i32 2, ptr @0)
   %inc = add i64 %ipt, 1
   ret i64 %inc
}
