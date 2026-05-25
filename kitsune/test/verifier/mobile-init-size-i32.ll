; If the mobile.init intrinsic call has 5 arguments, the size argument must be
; a 32-bit integer
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: call '{{.+}}' requires an additional size argument of type 'i32'

define void @f(ptr addrspace(67) %buf, i64 %n) {
  call void(i32, ptr addrspace(67), i64, ptr, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, ptr null, float 16.0000)
  ret void
}
