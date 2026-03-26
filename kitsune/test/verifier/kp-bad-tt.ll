; The tapir target that generates a global variable containing kernel properties
; must be one that generates embedded bitcode. It is unlikely that the serial
; tapir target will ever generate embedded bitcode.
;
;
; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: invalid value for 'kernel.properties' attribute. Tapir target does not generate embedded bitcode

@0 = constant { i64, i64, i64, i64 } zeroinitializer, !kit.gv !0

!0 = distinct !{!0, !1}
!1 = !{!"kit.gv.kernel.properties", i32 1, !"bristol"}
