; The value of the device.code attribute on a global variable must be a tapir
; target that generates embedded bitcode. It is unlikely that the serial tapir
; target will ever generate embedded bitcode.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: attribute 'kit.gv.device.code': invalid value
; CHECK-SAME: Tapir target does not generate embedded bitcode

@0 = constant [0 x i8] zeroinitializer, !kit.gv !0

!0 = distinct !{!0, !1}
!1 = !{!"kit.gv.device.code", i32 1}
