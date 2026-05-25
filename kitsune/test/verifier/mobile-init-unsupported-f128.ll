; The initializer in a call to kit.mobile.init cannot be fp128.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: unsupported initializer type 'fp128' in call '{{.+}}'

define void @f(ptr addrspace(67) %buf, i64 %n, fp128 %init) {
  call void(i32, ptr addrspace(67), i64, fp128, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, fp128 %init)
  ret void
}
