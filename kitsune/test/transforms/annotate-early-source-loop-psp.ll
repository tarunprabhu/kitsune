; Check that the kit-annotate-early pass correctly annotates loop nests that
; consist of a non-tapir loop nested between two tapir loops:
;
; forall (i ...)
;   for (j ...)
;     forall (k ...)
;
; RUN: opt -passes="kit-annotate-early" -S %s \
; RUN:     | FileCheck %s

; CHECK-LABEL: @f
;
; CHECK: br label %[[HI:.+]]
; CHECK: [[HI]]:
; CHECK: %i = phi i64 {{.+}} !kit.inst ![[I1:[0-9]+]]
; CHECK: detach {{.+}} !kit.inst ![[I2:[0-9]+]]
; CHECK: tail call {{.+}} !kit.inst ![[I3:[0-9]+]]
; CHECK: br label %[[HJ:.+]], !kit.inst ![[I4:[0-9]+]]
;
; CHECK: [[HJ]]:
; CHECK: %j = phi i64 {{.+}} ]{{$}}
; CHECK: br label %[[HK:.+]]{{$}}
;
; CHECK: [[HK]]:
; CHECK: %k = phi i64 {{.+}} !kit.inst ![[K1:[0-9]+]]
; CHECK: detach {{.+}} !kit.inst ![[K2:[0-9]+]]
; CHECK: reattach {{.+}} !kit.inst ![[K3:[0-9]+]]
; CHECK: add {{.+}} !kit.inst ![[K4:[0-9]+]]
; CHECK: icmp eq {{.+}} !kit.inst ![[K5:[0-9]+]]
; CHECK: br i1 {{.+}} !llvm.loop ![[LK:[0-9]+]], !kit.inst ![[K6:[0-9]+]]
;
; CHECK: add i64 %j, 1{{$}}
; CHECK: br i1 {{.+}} !llvm.loop ![[LJ:[0-9]+]]{{$}}
;
; CHECK: reattach {{.+}} !kit.inst ![[I5:[0-9]+]]
; CHECK: add {{.+}} !kit.inst ![[I6:[0-9]+]]
; CHECK: icmp eq {{.+}} !kit.inst ![[I7:[0-9]+]]
; CHECK: br i1 {{.+}} !llvm.loop ![[LI:[0-9]+]], !kit.inst ![[I8:[0-9]+]]

define dso_local void @f(i64 %m, i64 %n, i64 %p, i64 %q) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.k = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  br label %for.k.header

for.k.header:
  %k = phi i64 [ 0, %for.j.header ], [ %inc.k, %for.k.latch ]
  detach within %syncreg.k, label %for.k.body, label %for.k.latch

for.k.body:
  reattach within %syncreg.k, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !3

for.k.exit:
  sync within %syncreg.k, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !2

for.j.exit:
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

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}
!2 = distinct !{!2}
!3 = distinct !{!3, !0}

; CHECK-DAG: ![[SLI:[0-9]+]] = !{!"kit.inst.source.loop", ![[LI]]}
; CHECK-DAG: ![[SLK:[0-9]+]] = !{!"kit.inst.source.loop", ![[LK]]}
;
; CHECK-DAG: ![[I1]] = distinct !{![[I1]]{{.*}}, ![[SLI]]
; CHECK-DAG: ![[I2]] = distinct !{![[I2]]{{.*}}, ![[SLI]]
; CHECK-DAG: ![[I3]] = distinct !{![[I3]]{{.*}}, ![[SLI]]
; CHECK-DAG: ![[I4]] = distinct !{![[I4]]{{.*}}, ![[SLI]]
; CHECK-DAG: ![[I5]] = distinct !{![[I5]]{{.*}}, ![[SLI]]
; CHECK-DAG: ![[I6]] = distinct !{![[I6]]{{.*}}, ![[SLI]]
; CHECK-DAG: ![[I7]] = distinct !{![[I7]]{{.*}}, ![[SLI]]
; CHECK-DAG: ![[I8]] = distinct !{![[I8]]{{.*}}, ![[SLI]]
;
; CHECK-DAG: ![[K1]] = distinct !{![[K1]]{{.*}}, ![[SLK]]
; CHECK-DAG: ![[K2]] = distinct !{![[K2]]{{.*}}, ![[SLK]]
; CHECK-DAG: ![[K3]] = distinct !{![[K3]]{{.*}}, ![[SLK]]
; CHECK-DAG: ![[K4]] = distinct !{![[K4]]{{.*}}, ![[SLK]]
; CHECK-DAG: ![[K5]] = distinct !{![[K5]]{{.*}}, ![[SLK]]
; CHECK-DAG: ![[K6]] = distinct !{![[K6]]{{.*}}, ![[SLK]]
