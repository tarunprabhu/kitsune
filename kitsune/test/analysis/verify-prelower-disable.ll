; If the -kit-no-verify-prelower option is provided, the pass is effectively
; disabled even if it has been explicitly requested. Normally, this module
; would raise an error.
;
; RUN: opt --tapir=nolo -passes='kit-verify-prelower' -kit-no-verify-prelower \
; RUN:     -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; RUN: opt -passes='kit-verify-prelower' -kit-no-verify-prelower \
; RUN:     -disable-output %s 2>&1 \
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
