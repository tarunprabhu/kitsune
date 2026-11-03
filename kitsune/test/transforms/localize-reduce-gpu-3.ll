; Check that reductions in a nested loop with a GPU-centric tapir target are
; localized correctly
;
; RUN: opt -passes='kit-reduce-localize' -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-SAME: %[[RES:[^,]+]]
; CHECK-SAME: %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK: %[[SR1:.+]] = call token @llvm.syncregion.start()
; CHECK: detach within %[[SR1]]
; CHECK: %[[SR2:.+]] = call token @llvm.syncregion.start()
; CHECK: detach within %[[SR2]]
; CHECK: %[[SR3:.+]] = call token @llvm.syncregion.start()
; CHECK: %[[I3:.+]] = phi i64
; CHECK: detach within %[[SR3]], label %[[BODY3:.+]], label %[[LATCH3:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY3]]:
; CHECK-NEXT: %[[LOCAL1:.+]] = alloca i64
; CHECK-NEXT: store i64 1, ptr %[[LOCAL1]]
; CHECK-NEXT: br label %[[HEADER4:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER4]]
; CHECK-NEXT: %[[I4:.+]] = phi i64
; CHECK-SAME: [ 0, %[[BODY3]] ]
; CHECK-SAME: [ %[[INC4:.+]], %[[LATCH4:.+]] ]
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 2
; CHECK-SAME: i32 7
; CHECK-SAME: ptr %[[LOCAL1]]
; CHECK-SAME: i64 %[[I4]]
; CHECK-SAME: i32 1
; CHECK-SAME: i32 8
; CHECK-SAME: i64 1
; CHECK-SAME: ptr @mul
; CHECK-NEXT: %[[INC4:.+]] = add i64 %[[I4]], 1
; CHECK-NEXT: %[[CMP4:.+]] = icmp eq i64 %[[INC4]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP4]], label %[[REATTACH3:.+]], label %[[HEADER4]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH3]]
; CHECK-NEXT: %[[V:.+]] = load i64, ptr %[[LOCAL1]]
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 2
; CHECK-SAME: i32 7
; CHECK-SAME: ptr %[[RES]]
; CHECK-SAME: i64 %[[V]]
; CHECK-SAME: i32 10
; CHECK-SAME: i32 8
; CHECK-SAME: i64 1
; CHECK-SAME: ptr @mul
; CHECK-NEXT: reattach within %[[SR3]], label %[[LATCH3]]
; CHECK: br i1
; CHECK-SAME: !llvm.loop ![[LOOP3:[0-9]+]]
; CHECK: reattach within %[[SR2]]
; CHECK: br i1
; CHECK-SAME: !llvm.loop ![[LOOP2:[0-9]+]]
; CHECK: reattach within %[[SR1]]
; CHECK: br i1
; CHECK-SAME: !llvm.loop ![[LOOP1:[0-9]+]]

; CHECK-DAG: ![[LOCALIZED:[0-9]+]] = !{!"tapir.loop.localized"}
; CHECK-DAG: ![[LOOP1]] = distinct !{![[LOOP1]], {{.*}} ![[LOCALIZED]]
; CHECK-DAG: ![[LOOP2]] = distinct !{![[LOOP2]], {{.*}} ![[LOCALIZED]]
; CHECK-DAG: ![[LOOP3]] = distinct !{![[LOOP3]], {{.*}} ![[LOCALIZED]]

define void @f(ptr %res, i64 %n) {
entry:
  %sr1 = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i1 = phi i64 [ 0, %entry ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %sr2 = call token @llvm.syncregion.start()
  br label %header2

header2:
  %i2 = phi i64 [ 0, %body1 ], [ %inc2, %latch2 ]
  detach within %sr2, label %body2, label %latch2

body2:
  %sr3 = call token @llvm.syncregion.start()
  br label %header3

header3:
  %i3 = phi i64 [ 0, %body2 ], [ %inc3, %latch3 ]
  detach within %sr3, label %body3, label %latch3

body3:
  br label %header4

header4:
  %i4 = phi i64 [ 0, %body3 ], [ %inc4, %header4 ]
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 2, i32 7, ptr %res, i64 %i4, i32 0, i32 8, i64 1, ptr @mul)
  %inc4 = add i64 %i4, 1
  %cmp4 = icmp eq i64 %inc4, %n
  br i1 %cmp4, label %reattach3, label %header4, !llvm.loop !10

reattach3:
  reattach within %sr3, label %latch3

latch3:
  %inc3 = add i64 %i3, 1
  %cmp3 = icmp eq i64 %inc3, %n
  br i1 %cmp3, label %sync3, label %header3, !llvm.loop !9

sync3:
  sync within %sr3, label %reattach2

reattach2:
  reattach within %sr2, label %latch2

latch2:
  %inc2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %inc2, %n
  br i1 %cmp2, label %sync2, label %header2, !llvm.loop !8

sync2:
  sync within %sr2, label %reattach1

reattach1:
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !7

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

declare void @mul(ptr, i64)

!0 = !{!"tapir.loop.target", i32 2}
!1 = !{!"tapir.loop.perfect.level", i32 1}
!2 = !{!"tapir.loop.perfect.level", i32 2}
!3 = !{!"tapir.loop.perfect.level", i32 3}
!4 = !{!"tapir.loop.perfect.depth", i32 3}
!5 = !{!"tapir.loop.reduction"}
!6 = !{!"tapir.loop.prepared"}
!7 = distinct !{!7, !0, !1, !4, !5, !6}
!8 = distinct !{!8, !0, !2, !5, !6}
!9 = distinct !{!9, !0, !3, !5, !6}
!10 = distinct !{!10}
