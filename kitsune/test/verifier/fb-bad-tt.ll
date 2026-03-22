; The value of the device.code attribute on a global variable must be a tapir
; target that generates embedded bitcode. It is unlikely that the serial tapir
; target will ever generate embedded bitcode.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: invalid value for 'device.code' attribute. Tapir target does not generate embedded bitcode

@0 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !0

!0 = !{i32 1}
