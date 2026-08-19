; Check that the correct error is emitted if the first argument of a reducer
; is not a pointer
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: type mismatch in reducer 'sum' at param '1'. Expected 'ptr'

declare void @sum(i32 %dest, i8 %v)

define void @f() {
  %r = alloca i64
  call void(i32, i32, ptr, i32, i8, i8, ptr, ...) @llvm.kit.reduce.0(i32 1, i32 0, ptr %r, i32 8, i8 19, i8 127, ptr @sum)
  ret void
}
