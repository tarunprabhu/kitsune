; Tapir instructions outside a loop are ok, but only if they have the
; "kit.inst.target" attribute with value 'serial'
;
; RUN: opt -passes='kit-verify-prelower' -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{^.+$}}

define void @f() {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  detach within %syncreg, label %body, label %exit, !kit.inst !0

body:
  reattach within %syncreg, label %exit, !kit.inst !1

exit:
  ret void
}

!0 = distinct !{!0, !2}
!1 = distinct !{!1, !2}
!2 = !{!"kit.inst.target", i32 1}
