; Check that the correct error is emitted if the initializer of an embedded
; bitcode global variable could not be parsed to generate a valid LLVM module.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: global with attribute 'kit.gv.bit.code': invalid initializer
; CHECK-SAME: Could not parse bitcode
; CHECK-NEXT: from global variable 'bc.cuda'
; CHECK: global with attribute 'kit.gv.bit.code': invalid initializer
; CHECK-SAME: Could not parse bitcode
; CHECK-NEXT: from global variable 'bc.hip'

@bc.cuda = constant [8 x i8] c"BC\C0\DE1234", !kit.gv !0
@fb.cuda = constant [0 x i8] zeroinitializer, !kit.gv !1
@bc.hip = constant [8 x i8] c"BC\C0\DE1234", !kit.gv !2
@fb.hip = constant [0 x i8] zeroinitializer, !kit.gv !3

!0 = distinct !{!0, !4}
!1 = distinct !{!1, !5}
!2 = distinct !{!2, !6}
!3 = distinct !{!3, !7}
!4 = !{!"kit.gv.bit.code", i32 2}
!5 = !{!"kit.gv.device.code", i32 2}
!6 = !{!"kit.gv.bit.code", i32 4}
!7 = !{!"kit.gv.device.code", i32 4}
