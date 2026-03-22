; Global variables with the device.code attribute can have zero initializers
;
; RUN: llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{.+}}

@fb.cuda = constant [1 x i8] zeroinitializer, !kit.gv.device.code !0
@fb.hip = constant [1 x i8] zeroinitializer, !kit.gv.device.code !1

!0 = !{i32 2}
!1 = !{i32 4}
