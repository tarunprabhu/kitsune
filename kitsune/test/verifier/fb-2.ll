; Only a single device code global is allowed for a given tapir target.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: too many embedded device code globals for tapir target 'cuda'

@bc.2 = constant [0 x i8] zeroinitializer, !kit.gv !0
@.bc.2 = constant [0 x i8] zeroinitializer, !kit.gv !1
@bc.4 = constant [0 x i8] zeroinitializer, !kit.gv !2

!0 = distinct !{!0, !3}
!1 = distinct !{!1, !3}
!2 = distinct !{!2, !4}
!3 = !{!"kit.gv.device.code", i32 2}
!4 = !{!"kit.gv.device.code", i32 4}
