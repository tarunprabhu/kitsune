; Functions without a body may have Kitsune-specific attributes. The verifier
; must always check these.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: attribute 'kit.func.kernel': not compatible with 'kit.func.device'
; CHECK-NEXT: from function 'f'
; CHECK-NEXT: attribute 'kit.func.device': not compatible with 'kit.func.kernel'
; CHECK-NEXT: from function 'f'

declare !kit.func !0 void @f()

!0 = distinct !{!0, !1, !2}
!1 = !{!"kit.func.kernel", i32 3}
!2 = !{!"kit.func.device"}
