; REQUIRES: kitsune-hip && kitsune-cuda
;
; If a top-level loop has the hip tapir target, check that the correct
; diagnostic is emitted when at least one subloop has the cuda tapir target.
;
; NOTE: This combination will likely never be supported, which is why it
; probably merits is own test.
;
; RUN: not opt --passes=kit-verify-prelower %s -disable-output 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: tapir targets 'cuda' and 'hip' are incompatible in GPU loop nest
; CHECK-NEXT: from loop 'inner.loop'
; CHECK-NEXT: from function 'pps'
; CHECK-NEXT: target on ancestor loop is 'hip'

define void @pps(i64 %m, i64 %n, i64 %p) {
entry:
  %sr.i = tail call token @llvm.syncregion.start()
  br label %header.i

header.i:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch.i ]
  detach within %sr.i, label %body.i, label %latch.i

body.i:
  %sr.j = tail call token @llvm.syncregion.start()
  br label %header.j

header.j:
  %j = phi i64 [ 0, %body.i ], [ %inc.j, %latch.j ]
  detach within %sr.j, label %body.j, label %latch.j

body.j:
  br label %body.k

body.k:
  %k = phi i64 [ 0, %body.j ], [ %inc.k, %body.k ]
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %exit.k, label %body.k, !llvm.loop !2

exit.k:
  reattach within %sr.j, label %latch.j

latch.j:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %exit.j, label %header.j, !llvm.loop !1

exit.j:
  sync within %sr.j, label %end.j

end.j:
  reattach within %sr.i, label %latch.i

latch.i:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %exit.i, label %header.i, !llvm.loop !0

exit.i:
  sync within %sr.i, label %end.i

end.i:
  ret void
}

!0 = distinct !{!0, !3, !5, !8, !10}
!1 = distinct !{!1, !4, !6, !9}
!2 = distinct !{!2, !7}
!3 = !{!"tapir.loop.target", i32 4}
!4 = !{!"tapir.loop.target", i32 2}
!5 = !{!"tapir.loop.name", !"outer.loop"}
!6 = !{!"tapir.loop.name", !"inner.loop"}
!7 = !{!"tapir.loop.name", !"serial.loop"}
!8 = !{!"tapir.loop.perfect.level", i32 1}
!9 = !{!"tapir.loop.perfect.level", i32 2}
!10 = !{!"tapir.loop.perfect.depth", i32 2}
