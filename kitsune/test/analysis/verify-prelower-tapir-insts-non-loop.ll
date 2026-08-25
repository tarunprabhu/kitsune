; Tapir instructions, in principle, may appear outside loops. However, Kitsune
; does not yet support this, so don't allow it.
;
; RUN: not opt -passes='kit-verify-prelower' %s -disable-output 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: tapir instructions outside tapir loops are not yet supported
; CHECK: tapir instructions outside tapir loops are not yet supported

define void @f() {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  detach within %syncreg, label %body, label %exit

body:
  reattach within %syncreg, label %exit

exit:
  ret void
}
