; Global variables with the kit_fb attribute must have an initializer.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: global with attribute 'kit.gv.device.code': missing required initializer
; CHECK-NEXT: from global variable 'fb.cuda'
; CHECK: global with attribute 'kit.gv.device.code': missing required initializer
; CHECK-NEXT: from global variable 'fb.hip'

@fb.cuda = external global [0 x i8], !kit.gv !0
@fb.hip = external global [0 x i8], !kit.gv !1

!0 = distinct !{!0, !2}
!1 = distinct !{!1, !3}
!2 = !{!"kit.gv.device.code", i32 2}
!3 = !{!"kit.gv.device.code", i32 4}
