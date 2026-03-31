; Check that the kit-annotate-early pass adds the source.loop correctly to
; imperfectly nested tapir loops.
;
; RUN: opt -passes="kit-annotate-early" -S %s \
; RUN:     | FileCheck %s

; CHECK-LABEL: @ppp
; CHECK: br label %[[HI:.+]]
; CHECK: [[HI]]:
; CHECK: %i = phi i64 {{.+}} !kit.inst ![[I1:[0-9]+]]
; CHECK: detach {{.+}} !kit.inst ![[I2:[0-9]+]]
; CHECK: tail call {{.+}} !kit.inst ![[I3:[0-9]+]]
; CHECK: tail call void @ext(i64 111), !kit.inst ![[I4:[0-9]+]]
; CHECK: br label %[[HJ:.+]], !kit.inst ![[I5:[0-9]+]]
;
; CHECK: [[HJ]]:
; CHECK: %j = phi i64 {{.+}} !kit.inst ![[J1:[0-9]+]]
; CHECK: detach {{.+}} !kit.inst ![[J2:[0-9]+]]
; CHECK: tail call {{.+}} !kit.inst ![[J3:[0-9]+]]
; CHECK: tail call void @ext(i64 222), !kit.inst ![[J4:[0-9]+]]
; CHECK: br label %[[HK:.+]], !kit.inst ![[J5:[0-9]+]]
;
; CHECK: [[HK]]:
; CHECK: %k = phi i64 {{.+}} !kit.inst ![[K1:[0-9]+]]
; CHECK: detach {{.+}} !kit.inst ![[K2:[0-9]+]]
; CHECK: reattach {{.+}} !kit.inst ![[K3:[0-9]+]]
; CHECK: add {{.+}} !kit.inst ![[K4:[0-9]+]]
; CHECK: icmp eq {{.+}} !kit.inst ![[K5:[0-9]+]]
; CHECK: br i1 {{.+}} !llvm.loop ![[LK:[0-9]+]], !kit.inst ![[K6:[0-9]+]]
;
; CHECK: sync {{.+}} !kit.inst ![[J6:[0-9]+]]
; CHECK: reattach {{.+}} !kit.inst ![[J7:[0-9]+]]
; CHECK: tail call void @ext(i64 333), !kit.inst ![[J8:[0-9]+]]
; CHECK: add {{.+}} !kit.inst ![[J9:[0-9]+]]
; CHECK: icmp eq {{.+}} !kit.inst ![[J10:[0-9]+]]
; CHECK: br i1 {{.+}} !llvm.loop ![[LJ:[0-9]+]], !kit.inst ![[J11:[0-9]+]]
;
; CHECK: sync {{.+}} !kit.inst ![[I5:[0-9]+]]
; CHECK: reattach {{.+}} !kit.inst ![[I6:[0-9]+]]
; CHECK: tail call void @ext(i64 444), !kit.inst ![[I7:[0-9]+]]
; CHECK: add {{.+}} !kit.inst ![[I8:[0-9]+]]
; CHECK: icmp eq {{.+}} !kit.inst ![[I9:[0-9]+]]
; CHECK: br i1 {{.+}} !llvm.loop ![[LI:[0-9]+]], !kit.inst ![[I10:[0-9]+]]
;
define void @ppp(i64 %m, i64 %n, i64 %p) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  tail call void @ext(i64 111)
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  %syncreg.k = tail call token @llvm.syncregion.start()
  tail call void @ext(i64 222)
  br label %for.k.header

for.k.header:
  %k = phi i64 [0, %for.j.body ], [ %inc.k, %for.k.latch ]
  detach within %syncreg.k, label %for.k.body, label %for.k.latch

for.k.body:
  reattach within %syncreg.k, label %for.k.latch

for.k.latch:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !3

for.k.exit:
  sync within %syncreg.k, label %for.k.end

for.k.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  tail call void @ext(i64 333)
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !2

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  tail call void @ext(i64 444)
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !1

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

declare void @ext(i64)

!0 = !{!"tapir.loop.target", i32 1}
!1 = !{!1, !0}
!2 = !{!2, !0}
!3 = !{!3, !0}

; CHECK-DAG: ![[SLI:[0-9]+]] = !{!"kit.inst.source.loop", ![[LI]]}
; CHECK-DAG: ![[SLJ:[0-9]+]] = !{!"kit.inst.source.loop", ![[LJ]]}
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
; CHECK-DAG: ![[I9]] = distinct !{![[I9]]{{.*}}, ![[SLI]]
; CHECK-DAG: ![[I10]] = distinct !{![[I10]]{{.*}}, ![[SLI]]
;
; CHECK-DAG: ![[J1]] = distinct !{![[J1]]{{.*}}, ![[SLJ]]
; CHECK-DAG: ![[J2]] = distinct !{![[J2]]{{.*}}, ![[SLJ]]
; CHECK-DAG: ![[J3]] = distinct !{![[J3]]{{.*}}, ![[SLJ]]
; CHECK-DAG: ![[J4]] = distinct !{![[J4]]{{.*}}, ![[SLJ]]
; CHECK-DAG: ![[J5]] = distinct !{![[J5]]{{.*}}, ![[SLJ]]
; CHECK-DAG: ![[J6]] = distinct !{![[J6]]{{.*}}, ![[SLJ]]
; CHECK-DAG: ![[J7]] = distinct !{![[J7]]{{.*}}, ![[SLJ]]
; CHECK-DAG: ![[J8]] = distinct !{![[J8]]{{.*}}, ![[SLJ]]
; CHECK-DAG: ![[J9]] = distinct !{![[J9]]{{.*}}, ![[SLJ]]
; CHECK-DAG: ![[J10]] = distinct !{![[J10]]{{.*}}, ![[SLJ]]
; CHECK-DAG: ![[J11]] = distinct !{![[J11]]{{.*}}, ![[SLJ]]
;
; CHECK-DAG: ![[K1]] = distinct !{![[K1]]{{.*}}, ![[SLK]]
; CHECK-DAG: ![[K2]] = distinct !{![[K2]]{{.*}}, ![[SLK]]
; CHECK-DAG: ![[K3]] = distinct !{![[K3]]{{.*}}, ![[SLK]]
; CHECK-DAG: ![[K4]] = distinct !{![[K4]]{{.*}}, ![[SLK]]
; CHECK-DAG: ![[K5]] = distinct !{![[K5]]{{.*}}, ![[SLK]]
; CHECK-DAG: ![[K6]] = distinct !{![[K6]]{{.*}}, ![[SLK]]
