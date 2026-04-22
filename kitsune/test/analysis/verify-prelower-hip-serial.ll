; REQUIRES: kitsune-hip
;
; If a top-level loop has the hip tapir target, check that the correct
; diagnostic is emitted when at least one subloop has a serial tapir target.
; This is a separate test because it is a case where this situation can be
; handled by serializing the loop, but we do not yet do so. Until that changes,
; this should raise an error.
;
; RUN: not opt --tapir=nolo --passes=kit-verify-prelower %s 2>&1 \
; RUN:     -disable-output \
; RUN:     | FileCheck %s
;
; CHECK: tapir targets 'serial' and 'hip' are incompatible in GPU loop nest
; CHECK-NEXT: from loop 'inner.loop'
; CHECK-NEXT: from function 'hip_serial'
; CHECK-NEXT: target on ancestor loop is 'hip'

define void @hip_serial(i64 %m, i64 %n, i64 %p) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  br label %for.k.body

for.k.body:
  %k = phi i64 [ 0, %for.j.body ], [ %inc.k, %for.k.body ]
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.body, !llvm.loop !2

for.k.exit:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !1

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!0 = distinct !{!0, !3, !5}
!1 = distinct !{!1, !4, !6}
!2 = distinct !{!2, !7}
!3 = !{!"tapir.loop.target", i32 4}
!4 = !{!"tapir.loop.target", i32 1}
!5 = !{!"tapir.loop.name", !"outer.loop"}
!6 = !{!"tapir.loop.name", !"inner.loop"}
!7 = !{!"tapir.loop.name", !"serial.loop"}
