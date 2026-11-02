; Check that a tapir reduction loop of depth 3 with a CPU-centric tapir target
; is prepared correctly.
;
; RUN: opt -passes=kit-prepare -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[RESULT:[^,]+]]
; CHECK-SAME: i64 %[[N1:[^,]+]]
; CHECK-SAME: i64 %[[N2:[^,]+]]
; CHECK-SAME: i64 %[[N3:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK: br label %[[GUARDT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[GUARDT:.+]]:
; CHECK-NEXT: %[[SYSTHRDS:.+]] = call i64 @llvm.kit.cpu.num.threads(i32 1024)
; CHECK-NEXT: %[[CTX:.+]] = call ptr @__kitrt_nest_begin(i64 %[[SYSTHRDS]], i32 3)
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
; CHECK: %[[SLOTB2:.+]] = alloca i64
; CHECK-NEXT: %[[SLOTE2:.+]] = alloca i64
; CHECK-NEXT: call void @__kitrt_nest_bounds(ptr %[[CTX]], i64 %[[T]], i64 0, i64 %[[N2]], i32 2, ptr %[[SLOTB2]], ptr %[[SLOTE2]])
; CHECK-NEXT: %[[B2:.+]] = load i64, ptr %[[SLOTB2]]
; CHECK-NEXT: %[[E2:.+]] = load i64, ptr %[[SLOTE2]]
; CHECK-NEXT: %[[CMP2:.+]] = icmp ult i64 %[[B2]], %[[E2]]
; CHECK-NEXT: br i1 %[[CMP2]], label %[[PH2:.+]], label %[[BYPASS2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH2]]:
; CHECK-NEXT: br label %[[HEADER2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER2]]:
; CHECK-NEXT: %[[I2:.+]] = phi i64
; CHECK-SAME: [ %[[B2]], %[[PH2]] ]
; CHECK-SAME: [ %[[INC2:.+]], %[[LATCH2:.+]] ]
; CHECK-NEXT: br label %[[BODY2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY2]]:
; CHECK: %[[SLOTB3:.+]] = alloca i64
; CHECK-NEXT: %[[SLOTE3:.+]] = alloca i64
; CHECK-NEXT: call void @__kitrt_nest_bounds(ptr %[[CTX]], i64 %[[T]], i64 0, i64 %[[N3]], i32 3, ptr %[[SLOTB3]], ptr %[[SLOTE3]])
; CHECK-NEXT: %[[B3:.+]] = load i64, ptr %[[SLOTB3]]
; CHECK-NEXT: %[[E3:.+]] = load i64, ptr %[[SLOTE3]]
; CHECK-NEXT: %[[CMP3:.+]] = icmp ult i64 %[[B3]], %[[E3]]
; CHECK-NEXT: br i1 %[[CMP3]], label %[[PH3:.+]], label %[[BYPASS3:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH3]]:
; CHECK-NEXT: br label %[[HEADER3:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER3]]:
; CHECK-NEXT: %[[I3:.+]] = phi i64
; CHECK-SAME: [ %[[B3]], %[[PH3]] ]
; CHECK-SAME: [ %[[INC3:.+]], %[[LATCH3:.+]] ]
; CHECK-NEXT: br label %[[BODY3:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY3]]:
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 1024
; CHECK-SAME: i32 7
; CHECK-SAME: ptr %[[RESULT]]
; CHECK-SAME: i64 %[[I3]]
; CHECK-SAME: i32 0
; CHECK-SAME: i32 8
; CHECK-SAME: i64 1
; CHECK-SAME: ptr @mul
; CHECK-NEXT: br label %[[LATCH3]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH3]]
; CHECK-NEXT: %[[INC3:.+]] = add i64 %[[I3]], 1
; CHECK-NEXT: %[[CMP3:.+]] = icmp eq i64 %[[INC3]], %[[E3]]
; CHECK-NEXT: br i1 %[[CMP3]], label %[[LOOPEXIT3:.+]], label %[[HEADER3]]
; CHECK-SAME: llvm.loop ![[LOOP3:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT3]]:
; CHECK-NEXT: br label %[[BYPASS3]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BYPASS3]]:
; CHECK-NEXT: br label %[[REATTACH2:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH2]]:
; CHECK-NEXT: br label %[[LATCH2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH2]]:
; CHECK-NEXT: %[[INC2:.+]] = add i64 %[[I2]], 1
; CHECK-NEXT: %[[CMP2:.+]] = icmp eq i64 %[[INC2]], %[[E2]]
; CHECK-NEXT: br i1 %[[CMP2]], label %[[LOOPEXIT2:.+]], label %[[HEADER2]]
; CHECK-SAME: !llvm.loop ![[LOOP2:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT2]]:
; CHECK-NEXT: br label %[[BYPASS2]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BYPASS2]]:
; CHECK-NEXT: br label %[[REATTACH1:.+]]
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
; CHECK-NEXT: [[SYNCT]]
; CHECK-NEXT: call void @__kitrt_nest_end(ptr %[[CTX]])
; CHECK-NEXT: sync within %[[SRT]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]
; CHECK-NEXT: ret void

; CHECK-DAG: ![[DEPTH:[0-9]+]] = !{!"tapir.loop.perfect.depth", i32 1}
; CHECK-DAG: ![[LEVEL:[0-9]+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[PREPARED:[0-9]+]] = !{!"tapir.loop.prepared"}
; CHECK-DAG: ![[REDUCTION:[0-9]+]] = !{!"tapir.loop.reduction"}
; CHECK-DAG: ![[TARGET:[0-9]+]] = !{!"tapir.loop.target", i32 1024}

; CHECK-DAG: ![[LOOPT]] = distinct !{![[LOOPT]], ![[TARGET]], ![[DEPTH]], ![[LEVEL]], ![[REDUCTION]], ![[PREPARED]]}
; CHECK-DAG: ![[LOOP1]] = distinct !{![[LOOP1]]}
; CHECK-DAG: ![[LOOP2]] = distinct !{![[LOOP2]]}
; CHECK-DAG: ![[LOOP3]] = distinct !{![[LOOP3]]}

define void @f(ptr %result, i64 %n1, i64 %n2, i64 %n3) {
entry:
  %sr1 = call token @llvm.syncregion.start()
  %cmp.n1 = icmp slt i64 0, %n1
  br i1 %cmp.n1, label %ph1, label %sync1

ph1:
  br label %header1

header1:
  %i1 = phi i64 [ 0, %ph1 ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %sr2 = call token @llvm.syncregion.start()
  %cmp.n2 = icmp slt i64 0, %n2
  br i1 %cmp.n2, label %ph2, label %sync2

ph2:
  br label %header2

header2:
  %i2 = phi i64 [ 0, %ph2 ], [ %inc2, %latch2 ]
  detach within %sr2, label %body2, label %latch2

body2:
  %sr3 = call token @llvm.syncregion.start()
  %cmp.n3 = icmp slt i64 0, %n3
  br i1 %cmp.n3, label %ph3, label %sync3

ph3:
  br label %header3

header3:
  %i3 = phi i64 [ 0, %ph3 ], [ %inc3, %latch3 ]
  detach within %sr3, label %body3, label %latch3

body3:
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1024, i32 7, ptr %result, i64 %i3, i32 0, i32 8, i64 1, ptr @mul)
  reattach within %sr3, label %latch3

latch3:
  %inc3 = add i64 %i3, 1
  %cmp3 = icmp eq i64 %inc3, %n3
  br i1 %cmp3, label %loopexit3, label %header3, !llvm.loop !6

loopexit3:
  br label %sync3

sync3:
  sync within %sr3, label %reattach2

reattach2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %n2
  br i1 %cmp2, label %loopexit2, label %header2, !llvm.loop !7

loopexit2:
  br label %sync2

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n1
  br i1 %cmp1, label %loopexit1, label %header1, !llvm.loop !8

loopexit1:
  br label %sync1

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

declare void @mul(ptr, i64)

!0 = !{!"tapir.loop.target", i32 1024}
!1 = !{!"tapir.loop.perfect.depth", i32 3}
!2 = !{!"tapir.loop.perfect.level", i32 3}
!3 = !{!"tapir.loop.perfect.level", i32 2}
!4 = !{!"tapir.loop.perfect.level", i32 1}
!5 = !{!"tapir.loop.reduction"}
!6 = distinct !{!6, !0, !2}
!7 = distinct !{!7, !0, !3}
!8 = distinct !{!8, !0, !1, !4, !5}
