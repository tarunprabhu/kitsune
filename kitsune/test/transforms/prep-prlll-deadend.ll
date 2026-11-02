; Check that a tapir loop containing an exit to a dead-end block is prepared
; correctly. While we don't allow early exits in parallel loops in the frontend,
; branches to dead-end blocks can occur as a result of certain optimization
; passes.
;
; RUN: opt -passes=kit-prepare -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[N1:[^,]+]]
; CHECK-SAME: i64 %[[B:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK: br label %[[GUARDT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[GUARDT]]:
; CHECK-NEXT: %[[SYSTHRDS:.+]] = call i64 @llvm.kit.cpu.num.threads(i32 512)
; CHECK-NEXT: %[[CTX:.+]] = call ptr @__kitrt_nest_begin(i64 %[[SYSTHRDS]], i32 1)
; CHECK-NEXT: %[[THRDS:.+]] = call i64 @__kitrt_nest_threads(ptr %[[CTX]])
; CHECK-NEXT: %[[SRT:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: br i1 true, label %[[PHT:.+]], label %[[BYPASST:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PHT]]:
; CHECK-NEXT: br label %[[HEADERT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADERT]]:
; CHECK-NEXT: %[[T:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PHT]] ]
; CHECK-SAME: [ %[[INCT:.+]], %[[LATCHT:.+]] ]
; CHECK-NEXT: detach within %[[SRT]], label %[[BODYT:.+]], label %[[LATCHT]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODYT]]:
; CHECK: %[[SLOTB1:.+]] = alloca i64
; CHECK-NEXT: %[[SLOTE1:.+]] = alloca i64
; CHECK-NEXT: call void @__kitrt_nest_bounds(ptr %[[CTX]], i64 %[[T]], i64 0, i64 %[[N1]], i32 1, ptr %[[SLOTB1]], ptr %[[SLOTE1]])
; CHECK-NEXT: %[[B1:.+]] = load i64, ptr %[[SLOTB1]]
; CHECK-NEXT: %[[E1:.+]] = load i64, ptr %[[SLOTE1]]
; CHECK-NEXT: %[[CMP1:.+]] = icmp ult i64 %[[B1]], %[[E1]]
; CHECK-NEXT: br i1 %[[CMP1]], label %[[PH1:.+]], label %[[BYPASS1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH1]]:
; CHECK-NEXT: br label %[[HEADER1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER1]]:
; CHECK-NEXT: %[[I1:.+]] = phi i64
; CHECK-SAME: [ %[[B1]], %[[PH1]] ]
; CHECK-SAME: [ %[[INC1:.+]], %[[LATCH1:.+]] ]
; CHECK-NEXT: br label %[[BODY1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY1]]:
; CHECK-NEXT: %[[CHK:.+]] = icmp eq i64 %[[I1]], %[[B]]
; CHECK-NEXT: br i1 %[[CHK]], label %[[DEADEND:.+]], label %[[REATTACH1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH1]]:
; CHECK-NEXT: br label %[[LATCH1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH1]]:
; CHECK-NEXT: %[[INC1:.+]] = add i64 %[[I1]], 1
; CHECK-NEXT: %[[CMP1:.+]] = icmp eq i64 %[[INC1]], %[[E1]]
; CHECK-NEXT: br i1 %[[CMP1]], label %[[LOOPEXIT1:.+]], label %[[HEADER1]]
; CHECK-SAME: !llvm.loop ![[LOOP1:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT1]]:
; CHECK-NEXT: br label %[[BYPASS1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BYPASS1]]:
; CHECK-NEXT: br label %[[REATTACHT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[DEADEND]]:
; CHECK-NEXT: unreachable
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACHT]]:
; CHECK-NEXT: reattach within %[[SRT]], label %[[LATCHT]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCHT]]:
; CHECK-NEXT: %[[INCT]] = add nuw i64 %[[T]], 1
; CHECK-NEXT: %[[CMPT:.+]] = icmp eq i64 %[[INCT]], %[[THRDS]]
; CHECK-NEXT: br i1 %[[CMPT]], label %[[LOOPEXITT:.+]], label %[[HEADERT]]
; CHECK-SAME: !llvm.loop ![[LOOPT:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXITT]]:
; CHECK-NEXT: br label %[[SYNCT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNCT]]:
; CHECK-NEXT: call void @__kitrt_nest_end(ptr %[[CTX]])
; CHECK-NEXT: sync within %[[SRT]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void

; CHECK-DAG: ![[DEPTH:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 1}
; CHECK-DAG: ![[LEVEL:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[PREPARED:[0-9]+]] = !{!"tapir.loop.prepared"}
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 512}

; CHECK-DAG: ![[LOOPT]] = distinct !{![[LOOPT]], ![[TARGET]], ![[DEPTH]], ![[LEVEL]], ![[PREPARED]]}
; CHECK-DAG: ![[LOOP1]] = distinct !{![[LOOP1]]}

define void @f(i64 %n1, i64 %b) {
entry:
  %sr1 = call token @llvm.syncregion.start()
  %cmpn = icmp slt i64 0, %n1
  br i1 %cmpn, label %ph1, label %sync1

ph1:
  br label %header1

header1:
  %i1 = phi i64 [ 0, %ph1 ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %chk = icmp eq i64 %i1, %b
  br i1 %chk, label %deadend, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n1
  br i1 %cmp1, label %loopexit1, label %header1, !llvm.loop !0

loopexit1:
  br label %sync1

sync1:
  sync within %sr1, label %exit

deadend:
  unreachable

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.target", i32 512}
!2 = !{!"tapir.loop.perfect.depth", i32 1}
!3 = !{!"tapir.loop.perfect.level", i32 1}
