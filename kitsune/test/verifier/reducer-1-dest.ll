; Check that the correct error is emitted if the first argument of a reducer
; is not a pointer
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: type mismatch in reducer 'sum' at param '1'. Expected 'ptr'

define void @sum(double %dest, float %v) {
  ret void
}

define void @f(ptr addrspace(67) %buf, i64 %n) {
  %r = alloca i64
  call void(i32, ptr, i32, ptr addrspace(67), i64, float, ptr, ...) @llvm.kit.reduce.1(i32 1, ptr %r, i32 8, ptr addrspace(67) %buf, i64 %n, float 0.000000e+00, ptr @sum)
  ret void
}
