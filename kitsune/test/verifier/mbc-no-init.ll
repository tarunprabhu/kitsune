; Global variables with the 'bit.code' attribute must have an initializer.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK-COUNT-2: missing initializer in global containing bitcode

@bc.2 = external global [0 x i8], !kit.gv.bit.code !0
@fb.2 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !0
@bc.4 = external global [0 x i8], !kit.gv.bit.code !1
@fb.4 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !1

!0 = !{i32 2}
!1 = !{i32 4}
