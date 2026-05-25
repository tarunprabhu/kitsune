; The initializer in a call to kit.mobile.init cannot be x86_fp80. This
; corresponds to `long  double` in C++, or `real(10)` in Fortran on x86.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: unsupported initializer type 'x86_fp80' in call '{{.+}}'

define void @f(ptr addrspace(67) %buf, i64 %n, x86_fp80 %init) {
  call void(i32, ptr addrspace(67), i64, x86_fp80, ...) @llvm.kit.mobile.init(i32 1, ptr addrspace(67) %buf, i64 %n, x86_fp80 %init)
  ret void
}
