; Check that the correct error is emitted if a reducer does not contain the
; minimum number of arguments.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: incorrect number of params in reducer 'sum'. Expected 2

define void @sum(ptr %dest) {
  ret void
}

define void @f() {
  %r = alloca i64
  call void(i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 1, ptr %r, i32 8, i64 11, i64 0, ptr @sum)
  ret void
}
