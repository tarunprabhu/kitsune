; Sync instructions without a corresponding tapir loop are ok.
;
; RUN: opt --tapir=nolo -passes='kit-verify-prelower' %s 2>&1 \
; RUN:     -disable-output \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{.+}}

define void @f() {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  sync within %syncreg, label %exit

exit:
  ret void
}
