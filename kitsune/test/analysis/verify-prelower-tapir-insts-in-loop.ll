; If a loop is not determined to be a tapir loop, warn if detach/reattach
; instructions are found within it.
;
; RUN: not opt -passes='kit-verify-prelower' %s -disable-output 2>&1 \
; RUN:     | FileCheck %s

; CHECK: tapir instructions outside tapir loops are not yet supported
;
; for (i ... m)
define void @f1(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header.i

header.i:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch.i ]
  detach within %syncreg, label %body.i, label %latch.i

body.i:
  reattach within %syncreg, label %latch.i

latch.i:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %sync.i, label %header.i, !llvm.loop !0

sync.i:
  sync within %syncreg, label %exit.i

exit.i:
  ret void
}

; CHECK: tapir instructions outside tapir loops are not yet supported
;
; forall (i ... m)
;   for (j ... n)
;     for (k ... p)
define void @f3(i64 %m, i64 %n, i64 %p) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %header.i

header.i:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch.i ]
  detach within %syncreg.i, label %body.i, label %latch.i

body.i:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br label %header.j

header.j:
  %j = phi i64 [ 0, %body.i ], [ %inc.j, %latch.j ]
  detach within %syncreg.j, label %body.j, label %latch.j

body.j:
  br label %body.k

body.k:
  %k = phi i64 [ 0, %body.j ], [ %inc.k, %body.k ]
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %exit.k, label %body.k, !llvm.loop !3

exit.k:
  reattach within %syncreg.j, label %latch.j

latch.j:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %exit.j, label %header.j, !llvm.loop !2

exit.j:
  sync within %syncreg.j, label %end.j

end.j:
  reattach within %syncreg.i, label %latch.i

latch.i:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %exit.i, label %header.i, !llvm.loop !1

exit.i:
  sync within %syncreg.i, label %end.i

end.i:
  ret void
}

!0 = distinct !{!0, !4}
!1 = distinct !{!1, !5, !7}
!2 = distinct !{!2, !6}
!3 = distinct !{!3}
!4 = !{!"loop.name", !"f1.loop.i"}
!5 = !{!"loop.name", !"f3.loop.i"}
!6 = !{!"loop.name", !"f3.loop.j"}
!7 = !{!"tapir.loop.target", i32 1024}
