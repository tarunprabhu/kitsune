; The initializer of global variables with the kit_kernel_props attribute must
; be either a constant struct or a zero.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: global with attribute 'kit.gv.kernel.properties': invalid initializer
; CHECK-SAME: Must be a constant struct or zero-initialized{{$}}
; CHECK-NEXT: from global variable '@0'

@0 = constant { i64, i64, i64, i64 } undef, !kit.gv !0

!0 = distinct !{!0, !1}
!1 = !{!"kit.gv.kernel.properties", i32 2, !"coventry"}
