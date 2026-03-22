; Global variables with the 'bit.code' attribute must have type [n x i8] where
; n must be a positive integer.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: incorrect type of global containing bitcode

@bc.2 = constant i64 11, !kit.gv.bit.code !0
@fb.2 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !0
@bc.4 = constant i32 17, !kit.gv.bit.code !1
@fb.4 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !1

!0 = !{i32 2}
!1 = !{i32 4}
