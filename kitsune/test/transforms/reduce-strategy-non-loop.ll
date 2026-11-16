; The kit-reduce-strategy pass does not process calls to @llvm.kit.reduce that
; are not in a tapir loop.
;
; RUN: opt -passes='kit-reduce-strategy' -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[RESULT:[^,]+]]
; CHECK-SAME: i64 %[[I:[^)]+]]
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 1024
; CHECK-SAME: i32 5
; CHECK-SAME: ptr %[[RESULT]]
; CHECK-SAME: i64 %[[I]]
; CHECK-SAME: i32 0
; CHECK-SAME: i32 8
; CHECK-SAME: i64 0
; CHECK-SAME: ptr @add
; CHECK-NEXT: ret void

define void @f(ptr %result, i64 %i) {
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1024, i32 5, ptr %result, i64 %i, i32 0, i32 8, i64 0, ptr @add)
  ret void
}

declare void @add(ptr, i64)
