; The kit.gpu.register.devcode.end intrinsic is only relevant for the cuda tapir
; target.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: TTID argument in call must be 'cuda'

define void @f() {
  call void @llvm.kit.gpu.register.devcode.end(i32 4, ptr null)
  ret void
}
