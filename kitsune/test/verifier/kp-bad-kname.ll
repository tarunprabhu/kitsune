; The name of the function in a global variable containing kernel properties
; cannot be an empty string.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: attribute 'kit.gv.kernel.properties': invalid value at index '1'
; CHECK-SAME: Kernel name cannot be empty

@0 = constant { i64, i64, i64, i64 } zeroinitializer, !kit.gv !0

!0 = distinct !{!0, !1}
!1 = !{!"kit.gv.kernel.properties", i32 2, !""}
