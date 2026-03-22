; A global variable cannot have both the device.code and kernel.properties
; attributes.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK-COUNT-2: Attributes 'device.code' and 'kernel.properties' are incompatible

@g.2 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !0, !kit.gv.kernel.properties !2
@g.4 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !1, !kit.gv.kernel.properties !2

!0 = !{i32 2}
!1 = !{i32 4}
!2 = !{!"kname"}
