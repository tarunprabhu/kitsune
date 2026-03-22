; The initializer of global variables containing the bit.code attribute cannot
; be zero
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK-COUNT-2: invalid initializer in global containing bitcode

@bc.2 = constant [1 x i8] zeroinitializer, !kit.gv.bit.code !0
@fb.2 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !0
@bc.4 = constant [1 x i8] zeroinitializer, !kit.gv.bit.code !1
@fb.4 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !1

!0 = !{i32 2}
!1 = !{i32 4}
