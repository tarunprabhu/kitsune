; If no extra parameters are passed in the reduce intrinsic, the reducer should
; accept exactly 2 parameters.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: incorrect number of params in reducer 'sum'. Expected 2

declare void @sum(ptr %dest, i8 %v, ptr %extra)

define void @f() {
  %r = alloca i64
  call void(i32, i32, ptr, i32, i8, i8, ptr, ...) @llvm.kit.reduce.0(i32 1, i32 0, ptr %r, i32 8, i8 19, i8 127, ptr @sum)
  ret void
}
