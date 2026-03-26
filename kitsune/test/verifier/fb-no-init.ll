; Global variables with the kit_fb attribute must have an initializer.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK-COUNT-2: missing initializer in global containing device code

@fb.cuda = external global [0 x i8], !kit.gv !0
@fb.hip = external global [0 x i8], !kit.gv !1

!0 = distinct !{!0, !2}
!1 = distinct !{!1, !3}
!2 = !{!"kit.gv.device.code", i32 2}
!3 = !{!"kit.gv.device.code", i32 4}
