; Check that the kit.reduce intrinsic with the `Naive` strategy is lowered
; correctly.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt -passes=emb-lower-intrinsics \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[RESULT:.[^,]+]]
; CHECK-SAME: i64 %[[V:[^)]+]]
; CHECK-NEXT: call void @min(ptr %[[RESULT]], i64 %[[V]])
; CHECK-NEXT: ret void

declare void @min(ptr, i64)

define void @f(ptr %result, i64 %v) {
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 4, i32 26, ptr %result, i64 %v, i32 1, i32 8, i64 0, ptr @min)
  ret void
}
