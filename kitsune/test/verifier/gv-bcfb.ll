; A global variable cannot have both the bit.code and device.code attributes.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK-COUNT-2: Attributes 'bit.code' and 'device.code' are incompatible

@g.2 = constant [8 x i8] zeroinitializer, !kit.gv.bit.code !0, !kit.gv.device.code !0
@g.4 = constant [8 x i8] zeroinitializer, !kit.gv.bit.code !1, !kit.gv.device.code !1

!0 = !{i32 2}
!1 = !{i32 4}
