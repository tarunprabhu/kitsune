; Check that the correct error is emitted if the initializer of an embedded
; bitcode global variable could not be parsed to generate a valid LLVM module.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK-COUNT-2: could not parse embedded bitcode

@bc.2 = constant [8 x i8] c"BC\C0\DE1234", !kit.gv.bit.code !0
@fb.2 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !0
@bc.4 = constant [8 x i8] c"BC\C0\DE1234", !kit.gv.bit.code !1
@fb.4 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !1

!0 = !{i32 2}
!1 = !{i32 4}
