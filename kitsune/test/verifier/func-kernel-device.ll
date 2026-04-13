; A function cannot have both the kernel and device attributes.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: attribute 'kit.func.kernel': not compatible with 'kit.func.device'
; CHECK-NEXT: from function 'f'
; CHECK-NEXT: attribute 'kit.func.device': not compatible with 'kit.func.kernel'
; CHECK-NEXT: from function 'f'

define void @f() !kit.func !0 {
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"kit.func.kernel"}
!2 = !{!"kit.func.device"}
