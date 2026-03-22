; It is ok to have a global variable with the kernel.properties attribute that
; does not have a corresponding embedded bitcode global. The global containing
; embedded bitcode will be removed after the fat binary is generated, but the
; global with the kernel properties will not.
;
; RUN: llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{.+}}

@0 = constant { i64, i64, i64, i64 } zeroinitializer, !kit.gv.kernel.properties !0

!0 = !{!"birmingham"}
