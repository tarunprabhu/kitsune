; Check that the kit.gpu.reduce intrinsic with the `Naive` strategy is lowered
; correctly.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt -passes=emb-lower-intrinsics \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[RESULT:.[^,]+]]
; CHECK-SAME: i64 %[[V:[^)]+]]
; CHECK-NEXT: call void @sum(ptr %[[RESULT]], i64 %[[V]])
; CHECK-NEXT: ret void

declare void @sum(ptr, i64)

define void @f(ptr %result, i64 %v) {
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 2, i32 1, ptr %result, i64 %v, i32 1, i32 8, i64 0, ptr @sum)
  ret void
}
