; Sync instructions without a corresponding tapir loop are ok.
;
; RUN: opt -passes='kit-verify-prelower' %s -disable-output 2>&1 \
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
