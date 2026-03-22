; The initializer of a global variable with the device.code attribute must be
; either a constant array or a zero.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK-COUNT-2: invalid initializer in global containing device code

@fb.cuda = constant [1 x i8] undef, !kit.gv.device.code !0
@fb.hip = constant [1 x i8] undef, !kit.gv.device.code !1

!0 = !{i32 2}
!1 = !{i32 4}
