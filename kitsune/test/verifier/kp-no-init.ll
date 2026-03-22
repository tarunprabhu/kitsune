; Global variables containing kernel properties must have an initializer.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: missing initializer in global containing kernel properties

@0 = external global { i64, i64, i64, i64 }, !kit.gv.kernel.properties !0

!0 = !{!"lancashire"}
