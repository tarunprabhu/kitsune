; Check that the correct error is emitted if the embedded bitcode global
; variable does not contain bitcode.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: global with attribute 'kit.gv.bit.code': invalid initializer
; CHECK-SAME: Does not contain bitcode
; CHECK-NEXT: from global variable 'bc.cuda'
; CHECK: global with attribute 'kit.gv.bit.code': invalid initializer
; CHECK-SAME: Does not contain bitcode
; CHECK-NEXT: from global variable 'bc.hip'

@bc.cuda = constant [4 x i8] c"BC\C0\DF", !kit.gv !0
@fb.cuda = constant [0 x i8] zeroinitializer, !kit.gv !1
@bc.hip = constant [4 x i8] c"BC\C0\DF", !kit.gv !2
@fb.hip = constant [0 x i8] zeroinitializer, !kit.gv !3

!0 = distinct !{!0, !4}
!1 = distinct !{!1, !5}
!2 = distinct !{!2, !6}
!3 = distinct !{!3, !7}
!4 = !{!"kit.gv.bit.code", i32 2}
!5 = !{!"kit.gv.device.code", i32 2}
!6 = !{!"kit.gv.bit.code", i32 4}
!7 = !{!"kit.gv.device.code", i32 4}
