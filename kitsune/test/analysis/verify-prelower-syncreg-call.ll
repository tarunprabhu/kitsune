; Check that the syncregion passed to the detach, reattach and sync instructions
; are the result of a call to the llvm.syncregion.start intrinsic.
;
; RUN: not opt -passes='kit-verify-prelower' %s -disable-output 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: syncregion is not the result of an intrinsic call
; CHECK-NEXT: from basic block 'for.i.sync'
; CHECK-NEXT: from function 'f1'
;
; CHECK: syncregion is not the result of an intrinsic call
; CHECK-NEXT: from basic block 'for.i.header'
; CHECK-NEXT: from function 'f1'
;
; CHECK: syncregion is not the result of an intrinsic call
; CHECK-NEXT: from basic block 'for.i.body'
; CHECK-NEXT: from function 'f1'
;
define void @f1(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %syncreg.cst = bitcast token %syncreg to token
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.cst, label %for.i.body, label %for.i.latch

for.i.body:
  reattach within %syncreg.cst, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.sync, label %for.i.header, !llvm.loop !0

for.i.sync:
  sync within %syncreg.cst, label %for.i.exit

for.i.exit:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.target", i32 1024}
!2 = !{!"loop.name", !"f1.loop.i"}
