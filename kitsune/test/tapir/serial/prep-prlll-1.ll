; Check that a tapir loop nest of depth 1 with the serial tapir target is
; prepared as expected.
;
; RUN: opt --tapir=serial --passes=kit-prepare -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[N1:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: br label %[[GUARDW:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[GUARDW]]:
; CHECK-NEXT: %[[SRW:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: br i1 true, label %[[PHW:.+]], label %[[BYPASSW:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PHW]]:
; CHECK-NEXT: %[[THRDS:.+]] = call i64 @llvm.kit.cpu.num.threads(i32 1)
; CHECK-NEXT: br label %[[HEADERW:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADERW]]:
; CHECK-NEXT: %[[THRD:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PHW]] ],
; CHECK-SAME: [ %[[INCW:.+]], %[[LATCHW:.+]] ]
; CHECK-NEXT: detach within %[[SRW]], label %[[GUARD1:.+]], label %[[LATCHW]]
; CHECK-SAME: !kit.inst ![[DETACH:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[GUARD1]]:
; CHECK-NEXT: %[[CMPN:.+]] = icmp ult i64 0, %[[N1]]
; CHECK-NEXT: br i1 %[[CMPN]], label %[[PH1:.+]], label %[[BYPASS1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH1]]:
; CHECK-NEXT: br label %[[HEADER1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER1]]:
; CHECK-NEXT: %[[I1:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH1]] ]
; CHECK-SAME: [ %[[INC1:.+]], %[[LATCH1:.+]] ]
; CHECK-NEXT: br label %[[BODY1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY1]]:
; CHECK-NEXT: br label %[[LATCH1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH1]]:
; CHECK-NEXT: %[[INC1:.+]] = add i64 %[[I1]], 1
; CHECK-NEXT: %[[CMP1:.+]] = icmp eq i64 %[[INC1]], %[[N1]]
; CHECK-NEXT: br i1 %[[CMP1]], label %[[LOOPEXIT1:.+]], label %[[HEADER1]]
; CHECK-SAME: !llvm.loop ![[LOOP1:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT1]]:
; CHECK-NEXT: br label %[[BYPASS1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BYPASS1]]:
; CHECK-NEXT: br label %[[REATTACHW:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACHW]]:
; CHECK-NEXT: reattach within %[[SRW]], label %[[LATCHW]]
; CHECK-SAME: kit.inst ![[REATTACH:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCHW]]:
; CHECK-NEXT: %[[INCW]] = add nuw i64 %[[THRD]], 1
; CHECK-NEXT: %[[CMPW:.+]] = icmp eq i64 %[[INCW]], %[[THRDS]]
; CHECK-NEXT: br i1 %[[CMPW]], label %[[LOOPEXITW:.+]], label %[[HEADERW]]
; CHECK-SAME: !llvm.loop ![[LOOPW:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXITW]]:
; CHECK-NEXT: br label %[[BYPASSW]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BYPASSW]]:
; CHECK-NEXT: sync within %[[SRW]], label %[[EXIT:[^,]+]]
; CHECK-SAME: !kit.inst ![[SYNC:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
;
; CHECK-DAG: ![[DEPTH:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 1}
; CHECK-DAG: ![[LEVEL:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[PREPARED:[0-9]+]] = !{!"tapir.loop.prepared"}
; CHECK-DAG: ![[TARGET_INST:[0-9]+]] = !{!"kit.inst.target", i32 1}
; CHECK-DAG: ![[TARGET_LOOP:[0-9]+]] = !{!"tapir.loop.target", i32 1}
;
; CHECK-DAG: ![[DETACH]] = distinct !{![[DETACH]], ![[TARGET_INST]]}
; CHECK-DAG: ![[REATTACH]] = distinct !{![[REATTACH]], ![[TARGET_INST]]}
; CHECK-DAG: ![[SYNC]] = distinct !{![[SYNC]], ![[TARGET_INST]]}
;
; CHECK-DAG: ![[LOOP1]] = distinct !{![[LOOP1]]}
; CHECK-DAG: ![[LOOPW]] = distinct !{![[LOOPW]], ![[TARGET_LOOP]], ![[DEPTH]], ![[LEVEL]], ![[PREPARED]]}

define void @f(i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  %cmpn = icmp ult i64 0, %n
  br i1 %cmpn, label %ph, label %bypass

ph:
  br label %header

header:
  %i = phi i64 [ 0, %ph ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %loopexit, label %header, !llvm.loop !0

loopexit:
  br label %bypass

bypass:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"tapir.loop.perfect.depth", i32 1}
!3 = !{!"tapir.loop.perfect.level", i32 1}
