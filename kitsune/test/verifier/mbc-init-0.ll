; The initializer of global variables containing the bit.code attribute cannot
; be zero
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK-COUNT-2: invalid initializer in global containing bitcode

@bc.2 = constant [1 x i8] zeroinitializer, !kit.gv !0
@fb.2 = constant [0 x i8] zeroinitializer, !kit.gv !1
@bc.4 = constant [1 x i8] zeroinitializer, !kit.gv !2
@fb.4 = constant [0 x i8] zeroinitializer, !kit.gv !3

!0 = distinct !{!0, !4}
!1 = distinct !{!1, !5}
!2 = distinct !{!2, !6}
!3 = distinct !{!3, !7}
!4 = !{!"kit.gv.bit.code", i32 2}
!5 = !{!"kit.gv.device.code", i32 2}
!6 = !{!"kit.gv.bit.code", i32 4}
!7 = !{!"kit.gv.device.code", i32 4}
