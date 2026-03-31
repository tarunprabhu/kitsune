; Check that the kit-annotate-early pass works as expected for perfectly nested
; tapir loops of varying depths.
;
; RUN: opt -passes="kit-annotate-early" -S %s \
; RUN:     | FileCheck %s

; CHECK-LABEL: @p
; CHECK: br label %[[PHI:.+]]
; CHECK: [[PHI]]:
; CHECK: %i = phi i64 {{.+}} !kit.inst ![[PI1:[0-9]+]]
; CHECK: detach {{.+}} !kit.inst ![[PI2:[0-9]+]]
; CHECK: reattach {{.+}} !kit.inst ![[PI3:[0-9]+]]
; CHECK: add {{.+}} !kit.inst ![[PI4:[0-9]+]]
; CHECK: icmp eq {{.+}} !kit.inst ![[PI5:[0-9]+]]
; CHECK: br i1 {{.+}} !llvm.loop ![[PLI:[0-9]+]], !kit.inst ![[PI6:[0-9]+]]
;
define void @p1(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.sync, label %for.i.header, !llvm.loop !1

for.i.sync:
  sync within %syncreg, label %for.i.exit

for.i.exit:
  ret void
}

; CHECK-LABEL: @pp
; CHECK: br label %[[PPHI:.+]]
; CHECK: [[PPHI]]:
; CHECK: %i = phi i64 {{.+}} !kit.inst ![[PPI1:[0-9]+]]
; CHECK: detach {{.+}} !kit.inst ![[PPI2:[0-9]+]]
; CHECK: tail call {{.+}} !kit.inst ![[PPI3:[0-9]+]]
; CHECK: br label %[[PPHJ:.+]], !kit.inst ![[PPI4:[0-9]+]]
;
; CHECK: [[PPHJ]]:
; CHECK: %j = phi i64 {{.+}} !kit.inst ![[PPJ1:[0-9]+]]
; CHECK: detach {{.+}} !kit.inst ![[PPJ2:[0-9]+]]
; CHECK: reattach {{.+}} !kit.inst ![[PPJ3:[0-9]+]]
; CHECK: add {{.+}} !kit.inst ![[PPJ4:[0-9]+]]
; CHECK: icmp eq {{.+}} !kit.inst ![[PPJ5:[0-9]+]]
; CHECK: br i1 {{.+}} !llvm.loop ![[PPLJ:[0-9]+]], !kit.inst ![[PPJ6:[0-9]+]]
;
; CHECK: sync {{.+}} !kit.inst ![[PPI5:[0-9]+]]
; CHECK: reattach {{.+}} !kit.inst ![[PPI6:[0-9]+]]
; CHECK: add {{.+}} !kit.inst ![[PPI7:[0-9]+]]
; CHECK: icmp eq {{.+}} !kit.inst ![[PPI8:[0-9]+]]
; CHECK: br i1 {{.+}} !llvm.loop ![[PPLI:[0-9]+]], !kit.inst ![[PPI9:[0-9]+]]
;
define void @pp(i64 %m, i64 %n) {
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

; CHECK-LABEL: @ppp
; CHECK: br label %[[PPPHI:.+]]
; CHECK: [[PPPHI]]:
; CHECK: %i = phi i64 {{.+}} !kit.inst ![[PPPI1:[0-9]+]]
; CHECK: detach {{.+}} !kit.inst ![[PPPI2:[0-9]+]]
; CHECK: tail call {{.+}} !kit.inst ![[PPPI3:[0-9]+]]
; CHECK: br label %[[PPPHJ:.+]], !kit.inst ![[PPPI4:[0-9]+]]
;
; CHECK: [[PPPHJ]]:
; CHECK: %j = phi i64 {{.+}} !kit.inst ![[PPPJ1:[0-9]+]]
; CHECK: detach {{.+}} !kit.inst ![[PPPJ2:[0-9]+]]
; CHECK: tail call {{.+}} !kit.inst ![[PPPJ3:[0-9]+]]
; CHECK: br label %[[PPPHK:.+]], !kit.inst ![[PPPJ4:[0-9]+]]
;
; CHECK: [[PPPHK]]:
; CHECK: %k = phi i64 {{.+}} !kit.inst ![[PPPK1:[0-9]+]]
; CHECK: detach {{.+}} !kit.inst ![[PPPK2:[0-9]+]]
; CHECK: reattach {{.+}} !kit.inst ![[PPPK3:[0-9]+]]
; CHECK: add {{.+}} !kit.inst ![[PPPK4:[0-9]+]]
; CHECK: icmp eq {{.+}} !kit.inst ![[PPPK5:[0-9]+]]
; CHECK: br i1 {{.+}} !llvm.loop ![[PPPLK:[0-9]+]], !kit.inst ![[PPPK6:[0-9]+]]
;
; CHECK: sync {{.+}} !kit.inst ![[PPPJ5:[0-9]+]]
; CHECK: reattach {{.+}} !kit.inst ![[PPPJ6:[0-9]+]]
; CHECK: add {{.+}} !kit.inst ![[PPPJ7:[0-9]+]]
; CHECK: icmp eq {{.+}} !kit.inst ![[PPPJ8:[0-9]+]]
; CHECK: br i1 {{.+}} !llvm.loop ![[PPPLJ:[0-9]+]], !kit.inst ![[PPPJ9:[0-9]+]]
;
; CHECK: sync {{.+}} !kit.inst ![[PPPI5:[0-9]+]]
; CHECK: reattach {{.+}} !kit.inst ![[PPPI6:[0-9]+]]
; CHECK: add {{.+}} !kit.inst ![[PPPI7:[0-9]+]]
; CHECK: icmp eq {{.+}} !kit.inst ![[PPPI8:[0-9]+]]
; CHECK: br i1 {{.+}} !llvm.loop ![[PPPLI:[0-9]+]], !kit.inst ![[PPPI9:[0-9]+]]
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
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !6

for.k.exit:
  sync within %syncreg.k, label %for.k.end

for.k.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !5

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !4

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}
!2 = distinct !{!2, !0}
!3 = distinct !{!3, !0}
!4 = distinct !{!4, !0}
!5 = distinct !{!5, !0}
!6 = distinct !{!6, !0}
!7 = distinct !{!7, !0}
!8 = distinct !{!8, !0}
!9 = distinct !{!9, !0}
!10 = distinct !{!10, !0}

; ------------------------------------------------------------------------------
; CHECK-DAG: ![[PSLI:[0-9]+]] = !{!"kit.inst.source.loop", ![[PLI]]}
;
; CHECK-DAG: ![[PI1]] = distinct !{![[PI1]]{{.*}}, ![[PSLI]]
; CHECK-DAG: ![[PI2]] = distinct !{![[PI2]]{{.*}}, ![[PSLI]]
; CHECK-DAG: ![[PI3]] = distinct !{![[PI3]]{{.*}}, ![[PSLI]]
; CHECK-DAG: ![[PI4]] = distinct !{![[PI4]]{{.*}}, ![[PSLI]]
; CHECK-DAG: ![[PI5]] = distinct !{![[PI5]]{{.*}}, ![[PSLI]]
; CHECK-DAG: ![[PI6]] = distinct !{![[PI6]]{{.*}}, ![[PSLI]]
;
; ------------------------------------------------------------------------------
; CHECK-DAG: ![[PPSLI:[0-9]+]] = !{!"kit.inst.source.loop", ![[PPLI]]}
; CHECK-DAG: ![[PPSLJ:[0-9]+]] = !{!"kit.inst.source.loop", ![[PPLJ]]}
;
; CHECK-DAG: ![[PPI1]] = distinct !{![[PPI1]]{{.*}}, ![[PPSLI]]
; CHECK-DAG: ![[PPI2]] = distinct !{![[PPI2]]{{.*}}, ![[PPSLI]]
; CHECK-DAG: ![[PPI3]] = distinct !{![[PPI3]]{{.*}}, ![[PPSLI]]
; CHECK-DAG: ![[PPI4]] = distinct !{![[PPI4]]{{.*}}, ![[PPSLI]]
; CHECK-DAG: ![[PPI5]] = distinct !{![[PPI5]]{{.*}}, ![[PPSLI]]
; CHECK-DAG: ![[PPI6]] = distinct !{![[PPI6]]{{.*}}, ![[PPSLI]]
; CHECK-DAG: ![[PPI7]] = distinct !{![[PPI7]]{{.*}}, ![[PPSLI]]
; CHECK-DAG: ![[PPI8]] = distinct !{![[PPI8]]{{.*}}, ![[PPSLI]]
; CHECK-DAG: ![[PPI9]] = distinct !{![[PPI9]]{{.*}}, ![[PPSLI]]
;
; CHECK-DAG: ![[PPJ1]] = distinct !{![[PPJ1]]{{.*}}, ![[PPSLJ]]
; CHECK-DAG: ![[PPJ2]] = distinct !{![[PPJ2]]{{.*}}, ![[PPSLJ]]
; CHECK-DAG: ![[PPJ3]] = distinct !{![[PPJ3]]{{.*}}, ![[PPSLJ]]
; CHECK-DAG: ![[PPJ4]] = distinct !{![[PPJ4]]{{.*}}, ![[PPSLJ]]
; CHECK-DAG: ![[PPJ5]] = distinct !{![[PPJ5]]{{.*}}, ![[PPSLJ]]
; CHECK-DAG: ![[PPJ6]] = distinct !{![[PPJ6]]{{.*}}, ![[PPSLJ]]
;
; ------------------------------------------------------------------------------
; CHECK-DAG: ![[PPPSLI:[0-9]+]] = !{!"kit.inst.source.loop", ![[PPPLI]]}
; CHECK-DAG: ![[PPPSLJ:[0-9]+]] = !{!"kit.inst.source.loop", ![[PPPLJ]]}
; CHECK-DAG: ![[PPPSLK:[0-9]+]] = !{!"kit.inst.source.loop", ![[PPPLK]]}
;
; CHECK-DAG: ![[PPPI1]] = distinct !{![[PPPI1]]{{.*}}, ![[PPPSLI]]
; CHECK-DAG: ![[PPPI2]] = distinct !{![[PPPI2]]{{.*}}, ![[PPPSLI]]
; CHECK-DAG: ![[PPPI3]] = distinct !{![[PPPI3]]{{.*}}, ![[PPPSLI]]
; CHECK-DAG: ![[PPPI4]] = distinct !{![[PPPI4]]{{.*}}, ![[PPPSLI]]
; CHECK-DAG: ![[PPPI5]] = distinct !{![[PPPI5]]{{.*}}, ![[PPPSLI]]
; CHECK-DAG: ![[PPPI6]] = distinct !{![[PPPI6]]{{.*}}, ![[PPPSLI]]
; CHECK-DAG: ![[PPPI7]] = distinct !{![[PPPI7]]{{.*}}, ![[PPPSLI]]
; CHECK-DAG: ![[PPPI8]] = distinct !{![[PPPI8]]{{.*}}, ![[PPPSLI]]
; CHECK-DAG: ![[PPPI9]] = distinct !{![[PPPI9]]{{.*}}, ![[PPPSLI]]
;
; CHECK-DAG: ![[PPPJ1]] = distinct !{![[PPPJ1]]{{.*}}, ![[PPPSLJ]]
; CHECK-DAG: ![[PPPJ2]] = distinct !{![[PPPJ2]]{{.*}}, ![[PPPSLJ]]
; CHECK-DAG: ![[PPPJ3]] = distinct !{![[PPPJ3]]{{.*}}, ![[PPPSLJ]]
; CHECK-DAG: ![[PPPJ4]] = distinct !{![[PPPJ4]]{{.*}}, ![[PPPSLJ]]
; CHECK-DAG: ![[PPPJ5]] = distinct !{![[PPPJ5]]{{.*}}, ![[PPPSLJ]]
; CHECK-DAG: ![[PPPJ6]] = distinct !{![[PPPJ6]]{{.*}}, ![[PPPSLJ]]
; CHECK-DAG: ![[PPPJ7]] = distinct !{![[PPPJ7]]{{.*}}, ![[PPPSLJ]]
; CHECK-DAG: ![[PPPJ8]] = distinct !{![[PPPJ8]]{{.*}}, ![[PPPSLJ]]
; CHECK-DAG: ![[PPPJ9]] = distinct !{![[PPPJ9]]{{.*}}, ![[PPPSLJ]]
;
; CHECK-DAG: ![[PPPK1]] = distinct !{![[PPPK1]]{{.*}}, ![[PPPSLK]]
; CHECK-DAG: ![[PPPK2]] = distinct !{![[PPPK2]]{{.*}}, ![[PPPSLK]]
; CHECK-DAG: ![[PPPK3]] = distinct !{![[PPPK3]]{{.*}}, ![[PPPSLK]]
; CHECK-DAG: ![[PPPK4]] = distinct !{![[PPPK4]]{{.*}}, ![[PPPSLK]]
; CHECK-DAG: ![[PPPK5]] = distinct !{![[PPPK5]]{{.*}}, ![[PPPSLK]]
; CHECK-DAG: ![[PPPK6]] = distinct !{![[PPPK6]]{{.*}}, ![[PPPSLK]]
