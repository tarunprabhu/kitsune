; The type of global variables with the device.code attribute must be [n x i8]
; where N is some positive integer
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK-COUNT-2: incorrect type of global containing device code

@fb.cuda = constant i256 zeroinitializer, !kit.gv.device.code !0
@fb.hip = constant i256 zeroinitializer, !kit.gv.device.code !1

!0 = !{i32 2}
!1 = !{i32 4}
