; REQUIRES: kitsune-hip
;
; Check that the verifier does not emit a diagnostic message for certain
; non-canonical loops in a tapir loop nest for the GPU.
;
; RUN: opt --tapir=nolo -passes='kit-verify-prelower' -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: error:

; Non-canonical, non-tapir loops should not raise an error.
;
; forall (i ...)
;   for (j ...)
define void @ps(i64 %m, i64 %n) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 10, %for.i.body ], [ %inc.j, %for.j.header ]
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.i.reattach, label %for.j.header, !llvm.loop !1

for.i.reattach:
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

; A non-canonical, tapir loop that is not perfectly nested is ok.
;
; forall (i ...)
;   expr
;   forall (j ...)
define void @pep(i64 %m, i64 %n) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  call void @ext(i64 %i)
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 10, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !3

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !2

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

declare void @ext(i64)

!0 = distinct !{!0, !4}
!1 = distinct !{!1}
!2 = distinct !{!2, !4}
!3 = distinct !{!3, !4}
!4 = !{!"tapir.loop.target", i32 4}
