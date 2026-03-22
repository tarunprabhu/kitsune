; Global variables with the kit_fb attribute must have an initializer.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK-COUNT-2: missing initializer in global containing device code

@fb.cuda = external global [0 x i8], !kit.gv.device.code !0
@fb.hip = external global [0 x i8], !kit.gv.device.code !1

!0 = !{i32 2}
!1 = !{i32 4}
