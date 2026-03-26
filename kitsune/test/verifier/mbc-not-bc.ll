; Check that the correct error is emitted if the embedded bitcode global
; variable does not contain bitcode.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: invalid data in global containing embedded bitcode

@bc.2 = constant [4 x i8] c"BC\C0\DF", !kit.gv !0
@fb.2 = constant [0 x i8] zeroinitializer, !kit.gv !1
@bc.4 = constant [4 x i8] c"BC\C0\DF", !kit.gv !2
@fb.4 = constant [0 x i8] zeroinitializer, !kit.gv !3

!0 = distinct !{!0, !4}
!1 = distinct !{!1, !5}
!2 = distinct !{!2, !6}
!3 = distinct !{!3, !7}
!4 = !{!"kit.gv.bit.code", i32 2}
!5 = !{!"kit.gv.device.code", i32 2}
!6 = !{!"kit.gv.bit.code", i32 4}
!7 = !{!"kit.gv.device.code", i32 4}
