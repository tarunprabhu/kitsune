; Check that the kit.identity intrinsic is lowered as expected.
;
; RUN: opt -passes=kit-lower-intrinsics-early -S %s | FileCheck %s

; CHECK: @[[G:.+]] = external global
@g = external global i32

; CHECK-LABEL: @farg
; CHECK-SAME: i64 %[[A:[^)]+]]
; CHECK-NEXT: ret i64 %[[A]]
define i64 @farg(i64 %a) {
  %1 = call i64 @llvm.kit.identity.i64(i64 %a)
  ret i64 %1
}

; CHECK-LABEL: @fglobal
; CHECK-NEXT: ret ptr @[[G]]
define ptr @fglobal() {
  %1 = call ptr @llvm.kit.identity.ptr(ptr @g)
  ret ptr %1
}

; CHECK-LABEL: @fconst
; CHECK-NEXT: ret i1 false
define i1 @fconst() {
  %1 = call i1 @llvm.kit.identity.i1(i1 false)
  ret i1 %1
}

; CHECK-LABEL: finst
; CHECK-NEXT: %[[CALL:.+]] = call float @ext()
; CHECK-NEXT: ret float %[[CALL]]
define float @finst() {
  %1 = call float @ext()
  %2 = call float @llvm.kit.identity.f32(float %1)
  ret float %2
}

declare float @ext()
