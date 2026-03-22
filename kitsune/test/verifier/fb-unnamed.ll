; Globals containing device code must be named.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK-COUNT-2: global containing device code does not have a name

@0 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !0
@1 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !1

!0 = !{i32 2}
!1 = !{i32 4}
