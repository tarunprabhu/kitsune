; Currently, multiple tapir targets are not supported, even if, as is the case
; here, one of them is `serial` and the other `pthreads` which should generally
; be a "safe" combination.
;
; When we do support multi-target execution, this test should be removed because
; the corresponding code in the kit-verify-prelower pass will also have been
; removed. Tests d.val (and invalid) combinations of tapir targets should be
; done in other tests.
;
; RUN: not opt --tapir=serial -passes='kit-verify-prelower' %s 2>&1 \
; RUN:     -disable-output \
; RUN:     | FileCheck %s
;
; CHECK: error: compiling with multiple tapir targets is not yet supported

; forall (i ...)    // serial
;   forall (j ...)  // pthreads
;     for (k ...)
define void @pps(i64 %m, i64 %n, i64 %p) {
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
  br i1 %cmp.k, label %exit.k, label %body.k, !llvm.loop !2

exit.k:
  reattach within %syncreg.j, label %latch.j

latch.j:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %exit.j, label %header.j, !llvm.loop !1

exit.j:
  sync within %syncreg.j, label %end.j

end.j:
  reattach within %syncreg.i, label %latch.i

latch.i:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %exit.i, label %header.i, !llvm.loop !0

exit.i:
  sync within %syncreg.i, label %end.i

end.i:
  ret void
}

!0 = distinct !{!0, !3, !5, !7}
!1 = distinct !{!1, !4, !6}
!2 = distinct !{!2}
!3 = !{!"tapir.loop.target", i32 1}
!4 = !{!"tapir.loop.target", i32 1024}
!5 = !{!"tapir.loop.perfect.level", i32 1}
!6 = !{!"tapir.loop.perfect.level", i32 2}
!7 = !{!"tapir.loop.perfect.depth", i32 2}
