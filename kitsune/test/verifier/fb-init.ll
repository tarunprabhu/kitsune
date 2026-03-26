; The initializer of a global variable with the device.code attribute must be
; either a constant array or a zero.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK-COUNT-2: invalid initializer in global containing device code

@fb.cuda = constant [1 x i8] undef, !kit.gv !0
@fb.hip = constant [1 x i8] undef, !kit.gv !1

!0 = distinct !{!0, !2}
!1 = distinct !{!1, !3}
!2 = !{!"kit.gv.device.code", i32 2}
!3 = !{!"kit.gv.device.code", i32 4}
