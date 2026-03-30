; A global variable cannot have both the device.code and kernel.properties
; attributes.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: attribute 'kit.gv.device.code': not compatible with 'kit.gv.kernel.properties'
; CHECK-NEXT: from global variable 'g.cuda'
; CHECK: attribute 'kit.gv.device.code': not compatible with 'kit.gv.kernel.properties'
; CHECK-NEXT: from global variable 'g.hip'

@g.cuda = constant [0 x i8] zeroinitializer, !kit.gv !0
@g.hip = constant [0 x i8] zeroinitializer, !kit.gv !1

!0 = distinct !{!0, !2, !3}
!1 = distinct !{!1, !4, !5}
!2 = !{!"kit.gv.device.code", i32 2}
!3 = !{!"kit.gv.kernel.properties", i32 2, !"corpus christi, cambridge"}
!4 = !{!"kit.gv.device.code", i32 4}
!5 = !{!"kit.gv.kernel.properties", i32 4, !"corpus christi, oxford"}
