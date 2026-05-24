; Check that the correct error is emitted if a reducer returns a value.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: reducer 'sum' must return void

define ptr @sum(ptr %dest, i8 %v) {
  ret ptr %dest
}

define void @f() {
  %r = alloca i64
  call void(i32, ptr, i32, i8, i8, ptr, ...) @llvm.kit.reduce.0(i32 1, ptr %r, i32 8, i8 19, i8 127, ptr @sum)
  ret void
}
