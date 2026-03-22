; The tapir target that generates a global variable containing kernel properties
; must be one that generates embedded bitcode. It is unlikely that the serial
; tapir target will ever generate embedded bitcode.
;
;
; FIXME: This fails because we do not have a tapir target with the kernel
; properties global. This will be fixed soon.
; XFAIL: *
;
; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: invalid value for 'kernel.properties' attribute. Tapir target does not generate embedded bitcode

@0 = constant { i64, i64, i64, i64 } zeroinitializer, !kit.gv.kernel.properties !0

!0 = !{!"bristol"}
