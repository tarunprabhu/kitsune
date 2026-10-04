; Check that the correct error is emitted if the first argument of a reducer
; is not a pointer
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: type mismatch between value being reduced and unit

declare void @sum(ptr %dest, i8 %v)

define void @f() {
  %r = alloca i64
  call void(i32, i32, ptr, i8, i32, i32, i16, ptr, ...) @llvm.kit.reduce(i32 1, i32 0, ptr %r, i8 19, i32 0, i32 8, i16 -128, ptr @sum)
  ret void
}
