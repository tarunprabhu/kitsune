; The initializer in a call to kit.mobile.init cannot be fp128.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: incorrect number of arguments to call '{{.+}}'. Expected 5. Got 6

define void @f(ptr addrspace(67) %buf, i64 %n) {
  call void(i32, ptr addrspace(67), i64, ptr, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, ptr null, i32 32, i8 127)
  ret void
}
