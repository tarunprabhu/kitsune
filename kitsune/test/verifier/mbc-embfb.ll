; The embedded bitcode in this file contains a global variable contain device
; code. This is not allowed.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: embedded module: cannot contain embedded device code
; CHECK-NEXT: from global variable 'fb.cuda'
; CHECK: embedded module: cannot contain embedded device code
; CHECK-NEXT: from global variable 'fb.hip'

@fb.cuda = constant [0 x i8] zeroinitializer, !kit.gv !0
@fb.hip = constant [0 x i8] zeroinitializer, !kit.gv !2

!0 = distinct !{!0, !1}
!1 = !{!"kit.gv.device.code", i32 2}
!2 = distinct !{!2, !3}
!3 = !{!"kit.gv.device.code", i32 4}
