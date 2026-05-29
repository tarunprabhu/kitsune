; The kit.runtime.set.y.axis.kernel.launch intrinsic is only relevant for the
; hip tapir target.
;
; RUN: not opt -passes=verify %s 2>&1 | FileCheck %s
;
; CHECK: TTID argument in call must be 'hip'

define void @f() {
  call void @llvm.kit.runtime.set.y.axis.kernel.launch(i32 2, i8 1)
  ret void
}
