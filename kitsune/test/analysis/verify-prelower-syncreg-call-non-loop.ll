; Check that the syncregion passed to the detach, reattach and sync instructions
; are the result of a call to the llvm.syncregion.start intrinsic.
;
; RUN: not opt -passes='kit-verify-prelower' %s -disable-output 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: syncregion is not the result of an intrinsic call
; CHECK-NEXT: from basic block 'sync'
; CHECK-NEXT: from function 'f1'
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
