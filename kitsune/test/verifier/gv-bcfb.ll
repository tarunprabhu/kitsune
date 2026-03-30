; A global variable cannot have both the bit.code and device.code attributes.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: attribute 'kit.gv.bit.code': not compatible with 'kit.gv.device.code'
; CHECK-NEXT: from global variable 'g.cuda'
; CHECK: attribute 'kit.gv.bit.code': not compatible with 'kit.gv.device.code'
; CHECK-NEXT: from global variable 'g.hip'

@g.cuda = constant [8 x i8] zeroinitializer, !kit.gv !0
@g.hip = constant [8 x i8] zeroinitializer, !kit.gv !1

!0 = distinct !{!0, !2, !4}
!1 = distinct !{!1, !3, !5}
!2 = !{!"kit.gv.bit.code", i32 2}
!3 = !{!"kit.gv.bit.code", i32 4}
!4 = !{!"kit.gv.device.code", i32 2}
!5 = !{!"kit.gv.device.code", i32 4}
