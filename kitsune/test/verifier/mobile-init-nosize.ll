; If the initializer in the mobile.init intrinsic call is a pointer, an
; additional size argument must be provided
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: call '{{.+}}' requires an additional size argument of type 'i32'

define void @f(ptr addrspace(67) %buf, i64 %n) {
  call void(i32, ptr addrspace(67), i64, ptr, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, ptr null)
  ret void
}
