; Check that the correct error is emitted if the first argument of a reducer
; is not a pointer
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: type mismatch between value being reduced and unit

declare void @sum(ptr %dest, i8 %v)

define void @f() {
  %r = alloca i64
  call void(i32, ptr, i32, i8, i16, ptr, ...) @llvm.kit.reduce.0(i32 1, ptr %r, i32 8, i8 19, i16 -128, ptr @sum)
  ret void
}
