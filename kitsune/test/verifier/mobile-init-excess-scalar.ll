; The initializer in a call to kit.mobile.init cannot be fp128.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: incorrect number of arguments to call '{{.+}}'. Expected 4. Got 6

define void @f(ptr addrspace(67) %buf, i64 %n) {
  call void(i32, ptr addrspace(67), i64, i32, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, i32 9, ptr null, i8 127)
  ret void
}
