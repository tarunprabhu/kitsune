; The extra arguments passed to the kit.reduce intrinsic must match the extra
; parameters accepted by the reducer.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: type mismatch in reducer 'sum' at param '3'. Expected 'ptr'

declare void @sum(ptr %dest, i8 %v, i32 %another)

define void @f() {
  %r = alloca i64
  call void(i32, ptr, i32, i8, i8, ptr, ...) @llvm.kit.reduce.0(i32 1, ptr %r, i32 8, i8 19, i8 127, ptr @sum, ptr null)
  ret void
}
