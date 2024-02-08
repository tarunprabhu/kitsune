; The initializer in a call to kit.mobile.init cannot be i128.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: unsupported initializer type 'i128' in call '{{.+}}'

define void @f(ptr addrspace(67) %buf, i64 %n) {
  call void(i32, ptr addrspace(67), i64, i128, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, i128 29213)
  ret void
}
