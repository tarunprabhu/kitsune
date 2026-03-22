; The initializer of global variables with the kit_kernel_props attribute can be
; zero.
;
; RUN: llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{.+}}

@0 = constant [1 x i8] zeroinitializer, !kit.gv.kernel.properties !0

!0 = !{!"ucl"}
