; Check that the correct error is emitted if a reducer returns a value.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: reducer 'sum' must return void

define ptr @sum(ptr %dest, float %v) {
  ret ptr %dest
}

define void @f(ptr addrspace(67) %buf, i64 %n) {
  %r = alloca i64
  call void(i32, ptr, i32, ptr addrspace(67), i64, float, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %r, i32 8, ptr addrspace(67) %buf, i64 %n, float 0.000000e+00, ptr @sum)
  ret void
}
