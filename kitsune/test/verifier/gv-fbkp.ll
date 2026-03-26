; A global variable cannot have both the device.code and kernel.properties
; attributes.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK-COUNT-2: Attributes 'device.code' and 'kernel.properties' are incompatible

@g.2 = constant [0 x i8] zeroinitializer, !kit.gv !0
@g.4 = constant [0 x i8] zeroinitializer, !kit.gv !1

!0 = distinct !{!0, !2, !3}
!1 = distinct !{!1, !4, !5}
!2 = !{!"kit.gv.device.code", i32 2}
!3 = !{!"kit.gv.kernel.properties", i32 2, !"corpus christi, cambridge"}
!4 = !{!"kit.gv.device.code", i32 4}
!5 = !{!"kit.gv.kernel.properties", i32 4, !"corpus christi, oxford"}
