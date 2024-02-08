; The number of extra arguments passed to the kit.reduce intrinsic must match
; the number of extra parameters accepted by the reducer.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: incorrect number of params in reducer 'sum'. Expected 4. Got 3

declare void @sum(ptr %dest, i8 %v, ptr %extra)

define void @f() {
  %r = alloca i64
  call void(i32, ptr, i32, i8, i8, ptr, ...) @llvm.kit.reduce.0(i32 1, ptr %r, i32 8, i8 19, i8 127, ptr @sum, ptr null, i32 11)
  ret void
}
