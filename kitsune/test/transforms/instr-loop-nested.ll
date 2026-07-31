; Instrumenting tapir loops that are nested within other tapir loops is not
; supported. But instrumenting tapir loops that are within regular loops is ok.
;
; RUN: opt -passes=kit-instrument -S %s \
; RUN:     --kit-instr=timer 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: cannot instrument tapir loop that is not a top-level loop
; CHECK-NEXT: from loop 'inner'
;
; CHECK: @[[PARALLEL:.+]] = private{{.*}} constant [9 x i8] c"parallel\00"
;
; CHECK-LABEL: @sp
; CHECK: phi
; CHECK: %[[EPOCH:.+]] = call ptr @__kittimer_start(ptr @[[PARALLEL]], i64 0)
; CHECK: detach
; CHECK: reattach
; CHECK: sync
; CHECK: call i64 @__kittimer_stop(ptr %[[EPOCH]])
; CHECK: ret void

; forall (...)
;   forall (...)
define void @pp(i64 %m, i64 %n, i64 %p) {
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

define void @sp(i64 %m, i64 %n, i64 %p) {
entry:
  br label %header.i

header.i:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch.i ]
  %syncreg.j = tail call token @llvm.syncregion.start()
  br label %header.j

header.j:
  %j = phi i64 [ 0, %header.i ], [ %inc.j, %latch.j ]
  detach within %syncreg.j, label %body.j, label %latch.j

body.j:
  reattach within %syncreg.j, label %latch.j

latch.j:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %exit.j, label %header.j, !llvm.loop !6

exit.j:
  sync within %syncreg.j, label %latch.i

latch.i:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %exit.i, label %header.i, !llvm.loop !5

exit.i:
  ret void
}

!0 = distinct !{!0, !2, !3}
!1 = distinct !{!1, !2, !4}
!2 = !{!"tapir.loop.target", i32 1}
!3 = !{!"tapir.loop.name", !"outer"}
!4 = !{!"tapir.loop.name", !"inner"}
!5 = distinct !{!5}
!6 = distinct !{!6, !2, !7}
!7 = !{!"tapir.loop.name", !"parallel"}
