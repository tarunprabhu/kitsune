; Check that the correct error is emitted if a reducer does not contain the
; minimum number of arguments.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: incorrect number of params in reducer 'sum'. Expected 2

declare void @sum(ptr %dest)

define void @f(ptr addrspace(67) %buf, i64 %n) {
  %r = alloca i64
  call void(i32, ptr, i32, ptr addrspace(67), i64, i64, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %r, i32 8, ptr addrspace(67) %buf, i64 %n, i64 0, ptr @sum)
  ret void
}
