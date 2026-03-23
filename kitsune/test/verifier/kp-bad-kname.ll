; The name of the function in a global variable containing kernel properties
; cannot be an empty string.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: invalid value of 'kernel.properties' attribute. Kernel name cannot be empty

@0 = constant { i64, i64, i64, i64 } zeroinitializer, !kit.gv.kernel.properties !0

!0 = !{i32 2, !""}
