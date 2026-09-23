; Check that the tapir loop annotator pass correctly annotates tapir loops whose
; parent is a non-tapir loop.
;
; RUN: opt -passes="kit-annotate-early" -S %s | FileCheck %s

; CHECK-LABEL: @sspp
; CHECK: !llvm.loop ![[SSPP_L:[0-9]+]]
; CHECK: !llvm.loop ![[SSPP_K:[0-9]+]]
; CHECK: !llvm.loop ![[SSPP_J:[0-9]+]]
; CHECK: !llvm.loop ![[SSPP_I:[0-9]+]]
;
; for (i ...)
;   for (j ...)
;     forall (k ...)
;       forall (j ...)
;         ...
;
define void @sspp(i64 %m, i64 %n, i64 %p, i64 %q) {
entry:
  br label %header.i

header.i:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch.i ]
  br label %header.j

header.j:
  %j = phi i64 [ 0, %header.i ], [ %inc.j, %latch.j ]
  %syncreg.k = call token @llvm.syncregion.start()
  br label %header.k

header.k:
  %k = phi i64 [ 0, %header.j ], [ %inc.k, %latch.k ]
  detach within %syncreg.k, label %body.k, label %latch.k

body.k:
  %syncreg.l = call token @llvm.syncregion.start()
  br label %header.l

header.l:
  %l = phi i64 [ 0, %body.k ], [ %inc.l, %latch.l ]
  detach within %syncreg.l, label %body.l, label %latch.l

body.l:
  reattach within %syncreg.l, label %latch.l

latch.l:
  %inc.l = add i64 %l, 1
  %cmp.l = icmp eq i64 %inc.l, %q
  br i1 %cmp.l, label %exit.l, label %header.l, !llvm.loop !1

exit.l:
  sync within %syncreg.l, label %end.l

end.l:
  reattach within %syncreg.k, label %latch.k

latch.k:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %exit.k, label %header.k, !llvm.loop !2

exit.k:
  sync within %syncreg.k, label %latch.j

latch.j:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %latch.i, label %header.j, !llvm.loop !3

latch.i:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %exit, label %header.i, !llvm.loop !4

exit:
  ret void
}

; CHECK-LABEL: @sspep
; CHECK: !llvm.loop ![[SSPEP_L:[0-9]+]]
; CHECK: !llvm.loop ![[SSPEP_K:[0-9]+]]
; CHECK: !llvm.loop ![[SSPEP_J:[0-9]+]]
; CHECK: !llvm.loop ![[SSPEP_I:[0-9]+]]
;
; for (i ...)
;   for (j ...)
;     forall (k ...)
;       expr
;       forall (j ...)
;         ...
;
define void @sspep(i64 %m, i64 %n, i64 %p, i64 %q) {
entry:
  br label %header.i

header.i:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch.i ]
  br label %header.j

header.j:
  %j = phi i64 [ 0, %header.i ], [ %inc.j, %latch.j ]
  %syncreg.k = call token @llvm.syncregion.start()
  br label %header.k

header.k:
  %k = phi i64 [ 0, %header.j ], [ %inc.k, %latch.k ]
  detach within %syncreg.k, label %body.k, label %latch.k

body.k:
  %syncreg.l = call token @llvm.syncregion.start()
  call void @ext(i64 %k)
  br label %header.l

header.l:
  %l = phi i64 [ 0, %body.k ], [ %inc.l, %latch.l ]
  detach within %syncreg.l, label %body.l, label %latch.l

body.l:
  reattach within %syncreg.l, label %latch.l

latch.l:
  %inc.l = add i64 %l, 1
  %cmp.l = icmp eq i64 %inc.l, %q
  br i1 %cmp.l, label %exit.l, label %header.l, !llvm.loop !5

exit.l:
  sync within %syncreg.l, label %end.l

end.l:
  reattach within %syncreg.k, label %latch.k

latch.k:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %exit.k, label %header.k, !llvm.loop !6

exit.k:
  sync within %syncreg.k, label %latch.j

latch.j:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %latch.i, label %header.j, !llvm.loop !7

latch.i:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %end, label %header.i, !llvm.loop !8

end:
  ret void
}

declare void @ext(i64)

!0 = distinct !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}
!2 = distinct !{!2, !0}
!3 = distinct !{!3}
!4 = distinct !{!4}
!5 = distinct !{!5, !0}
!6 = distinct !{!6, !0}
!7 = distinct !{!7}
!8 = distinct !{!8}

;-------------------------------------------------------------------------------
;
; CHECK-DAG: ![[D2:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 2}
; CHECK-DAG: ![[L1:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[L2:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 2}
;
;-------------------------------------------------------------------------------
;
; for (i ...)
;   for (j ...)
;     forall (k ...)
;       forall (l ...)
;
; CHECK-DAG: ![[SSPP_I]] = distinct !{![[SSPP_I]]}
; CHECK-DAG: ![[SSPP_J]] = distinct !{![[SSPP_J]]}
; CHECK-DAG: ![[SSPP_K]] = distinct !{![[SSPP_K]], {{.+}}, ![[L1]], ![[D2]]}
; CHECK-DAG: ![[SSPP_L]] = distinct !{![[SSPP_L]], {{.+}}, ![[L2]]}
;
;-------------------------------------------------------------------------------
;
; for (i ...)
;   for (j ...)
;     forall (k ...)
;       expr
;       forall (l ...)
;         ...
;
; CHECK-DAG: ![[SSPEP_I]] = distinct !{![[SSPEP_I]]}
; CHECK-DAG: ![[SSPEP_J]] = distinct !{![[SSPEP_J]]}
; CHECK-DAG: ![[SSPEP_K]] = distinct !{![[SSPEP_K]], {{.+}}, ![[L1]], ![[D2]]}
; CHECK-DAG: ![[SSPEP_L]] = distinct !{![[SSPEP_L]], {{.+}}, ![[L2]]}
;
;-------------------------------------------------------------------------------
