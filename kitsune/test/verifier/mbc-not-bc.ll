; Check that the correct error is emitted if the embedded bitcode global
; variable does not contain bitcode.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: invalid data in global containing embedded bitcode

@bc.2 = constant [4 x i8] c"BC\C0\DF", !kit.gv.bit.code !0
@fb.2 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !0
@bc.4 = constant [4 x i8] c"BC\C0\DF", !kit.gv.bit.code !1
@fb.4 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !1

!0 = !{i32 2}
!1 = !{i32 4}
