; Multiple device code globals are allowed in a host module as long as they are
; from distinct tapir targets.
;
; RUN: llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{.+}}

@bc.2 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !0
@bc.4 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !1

!0 = !{i32 2}
!1 = !{i32 4}
