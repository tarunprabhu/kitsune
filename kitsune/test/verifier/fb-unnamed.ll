; Globals containing device code must be named.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK-COUNT-2: global with attribute 'kit.gv.device.code': missing required name

@0 = constant [0 x i8] zeroinitializer, !kit.gv !0
@1 = constant [0 x i8] zeroinitializer, !kit.gv !1

!0 = distinct !{!0, !2}
!1 = distinct !{!1, !3}
!2 = !{!"kit.gv.device.code", i32 2}
!3 = !{!"kit.gv.device.code", i32 4}
