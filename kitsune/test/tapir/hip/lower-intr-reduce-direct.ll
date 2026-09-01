; Check that the gpu.reduce.direct intrinsic is lowered correctly.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt -passes=emb-lower-intrinsics \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: atomicrmw umax ptr %result, i64 %v monotonic
; CHECK-NEXT: ret void

declare void @min(ptr, i32)

define void @f(ptr %result, i64 %v) {
  call void(i32, i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.gpu.reduce.direct(i32 4, i32 26, ptr %result, i32 8, i64 %v, i64 0, ptr @min)
  ret void
}
