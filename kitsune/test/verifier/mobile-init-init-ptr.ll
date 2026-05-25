; If the mobile.init intrinsic call has 5 arguments, the initializer must be a
; pointer.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: initializer argument in call '{{.+}}' must be a pointer

define void @f(ptr addrspace(67) %buf, i64 %n) {
  call void(i32, ptr addrspace(67), i64, i8, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, i8 1, i32 8)
  ret void
}
