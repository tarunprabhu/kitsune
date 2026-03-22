; Only a single device code global is allowed for a given tapir target.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: too many embedded device code globals for tapir target 'cuda'

@bc.2 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !0
@.bc.2 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !0
@bc.4 = constant [0 x i8] zeroinitializer, !kit.gv.device.code !1

!0 = !{i32 2}
!1 = !{i32 4}
