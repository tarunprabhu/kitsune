; The extra arguments passed to the kit.reduce intrinsic must match the extra
; parameters accepted by the reducer.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: type mismatch in reducer 'sum' at param '3'. Expected 'ptr'

declare void @sum(ptr %dest, i64 %v, i32 %another)

define void @f(ptr addrspace(67) %buf, i64 %n) {
  %r = alloca i64
  call void(i32, ptr, i32, ptr addrspace(67), i64, i64, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %r, i32 8, ptr addrspace(67) %buf, i64 %n, i64 0, ptr @sum, ptr null)
  ret void
}
