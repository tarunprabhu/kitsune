; The number of extra arguments passed to the kit.reduce intrinsic must match
; the number of extra parameters accepted by the reducer.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: incorrect number of params in reducer 'sum'. Expected 4. Got 3

declare void @sum(ptr %dest, i64 %v, ptr %extra)

define void @f(ptr addrspace(67) %buf, i64 %n) {
  %r = alloca i64
  call void(i32, ptr, i32, ptr addrspace(67), i64, i64, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %r, i32 8, ptr addrspace(67) %buf, i64 %n, i64 0, ptr @sum, ptr null, i32 10)
  ret void
}
