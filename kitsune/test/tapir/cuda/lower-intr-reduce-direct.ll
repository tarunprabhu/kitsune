; Check that the gpu.reduce.direct intrinsic is lowered correctly.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt -passes=emb-lower-intrinsics \
; RUN:     | %kit-mbc -S -o - \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: atomicrmw add ptr %result, i32 %v monotonic
; CHECK-NEXT: ret void

declare void @sum(ptr, i32)

define void @f(ptr %result, i32 %v) {
  call void(i32, i32, ptr, i32, i32, i32, ptr, ...) @llvm.kit.gpu.reduce.direct(i32 2, i32 5, ptr %result, i32 4, i32 %v, i32 0, ptr @sum)
  ret void
}
