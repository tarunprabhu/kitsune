; If a tapir target is not set, the kit-verify-prelower pass will not run, even
; even if it has been explicitly requested. This module will otherwise cause an
; error to be raised.
;
; RUN: opt -passes='kit-verify-prelower' -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{.+}}
;
define void @f1() {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %syncreg.cst = bitcast token %syncreg to token
  br label %sync

sync:
  sync within %syncreg.cst, label %exit

exit:
  ret void
}
