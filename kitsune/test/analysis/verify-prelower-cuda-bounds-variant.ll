; The loop bounds of perfectly nested tapir loops with the 'cuda' tapir target
; must be invariant with respect to the root.
;
; RUN: not opt -passes='kit-verify-prelower' --tapir=nolo \
; RUN:     -disable-output %s 2>&1 \
; RUN:     | FileCheck %s

; CHECK: bounds of GPU loop must be invariant in the nest
; CHECK-NEXT: from loop 'pv.loop.j'
; CHECK-NEXT: from function 'pv'
; CHECK-NEXT: root of tapir loop nest is 'pv.loop.i'
;
; forall (i ...)
;   forall (j ...)
;
define void @pv(i64 %m, i64 %n) {
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
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %i
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !2

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !1

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

; CHECK: bounds of GPU loop must be invariant in the nest
; CHECK-NEXT: from loop 'pvv.loop.j'
; CHECK-NEXT: from function 'pvv'
; CHECK-NEXT: root of tapir loop nest is 'pvv.loop.i'
;
; CHECK: bounds of GPU loop must be invariant in the nest
; CHECK-NEXT: from loop 'pvv.loop.k'
; CHECK-NEXT: from function 'pvv'
; CHECK-NEXT: root of tapir loop nest is 'pvv.loop.i'
;
; forall (i ...)
;   forall (j ... i)
;     forall (k ... j)
;
define void @pvv(i64 %m, i64 %n, i64 %p) {
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
  %syncreg.k = tail call token @llvm.syncregion.start()
  br label %for.k.header

for.k.header:
  %k = phi i64 [0, %for.j.body ], [ %inc.k, %for.k.latch ]
  detach within %syncreg.k, label %for.k.body, label %for.k.latch

for.k.body:
  reattach within %syncreg.k, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %i
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !5

for.k.exit:
  sync within %syncreg.k, label %for.k.end

for.k.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %j
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !4

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !3

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

; CHECK: bounds of GPU loop must be invariant in the nest
; CHECK-NEXT: from loop 'ppv.loop.k'
; CHECK-NEXT: from function 'ppv'
; CHECK-NEXT: root of tapir loop nest is 'ppv.loop.i'
;
; forall (i ...)
;   forall (j ... )
;     forall (k ... i)
;
define void @ppv(i64 %m, i64 %n, i64 %p) {
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
  %syncreg.k = tail call token @llvm.syncregion.start()
  br label %for.k.header

for.k.header:
  %k = phi i64 [0, %for.j.body ], [ %inc.k, %for.k.latch ]
  detach within %syncreg.k, label %for.k.body, label %for.k.latch

for.k.body:
  reattach within %syncreg.k, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %i
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !8

for.k.exit:
  sync within %syncreg.k, label %for.k.end

for.k.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !7

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !6

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

; CHECK: bounds of GPU loop must be invariant in the nest
; CHECK-NEXT: from loop 'pvp.loop.j'
; CHECK-NEXT: from function 'pvp'
; CHECK-NEXT: root of tapir loop nest is 'pvp.loop.i'
;
; forall (i ...)
;   forall (j ... i)
;     forall (k ... )
;
define void @pvp(i64 %m, i64 %n, i64 %p) {
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
  %syncreg.k = tail call token @llvm.syncregion.start()
  br label %for.k.header

for.k.header:
  %k = phi i64 [0, %for.j.body ], [ %inc.k, %for.k.latch ]
  detach within %syncreg.k, label %for.k.body, label %for.k.latch

for.k.body:
  reattach within %syncreg.k, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !11

for.k.exit:
  sync within %syncreg.k, label %for.k.end

for.k.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %i
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !10

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !9

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 2}
!1 = distinct !{!1, !0, !12}
!2 = distinct !{!2, !0, !13}
!3 = distinct !{!3, !0, !14}
!4 = distinct !{!4, !0, !15}
!5 = distinct !{!5, !0, !16}
!6 = distinct !{!6, !0, !17}
!7 = distinct !{!7, !0, !18}
!8 = distinct !{!8, !0, !19}
!9 = distinct !{!9, !0, !20}
!10 = distinct !{!10, !0, !21}
!11 = distinct !{!11, !0, !22}
!12 = !{!"loop.name", !"pv.loop.i"}
!13 = !{!"loop.name", !"pv.loop.j"}
!14 = !{!"loop.name", !"pvv.loop.i"}
!15 = !{!"loop.name", !"pvv.loop.j"}
!16 = !{!"loop.name", !"pvv.loop.k"}
!17 = !{!"loop.name", !"ppv.loop.i"}
!18 = !{!"loop.name", !"ppv.loop.j"}
!19 = !{!"loop.name", !"ppv.loop.k"}
!20 = !{!"loop.name", !"pvp.loop.i"}
!21 = !{!"loop.name", !"pvp.loop.j"}
!22 = !{!"loop.name", !"pvp.loop.k"}
