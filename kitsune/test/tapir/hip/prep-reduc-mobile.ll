; If the destination of a reduction is in Kitsune's mobile address space, the
; loop is not transformed.
;
; By default, shadow memory is allocated using UVM.
;
; RUN: opt --passes=kit-prepare -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-SAME: ptr addrspace(67) %[[RES67:[^,]+]]
; CHECK-SAME: i64 %[[N1:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SR1:.+]] = tail call token @llvm.syncregion.start()
; CHECK-NEXT: %[[CMPN1:.+]] = icmp
; CHECK-NEXT: br i1 %[[CMPN1]], label %[[PH1:.+]], label %[[SYNC1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[PH1]]:
; CHECK-NEXT: br label %[[HEADER1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER1]]:
; CHECK-NEXT: %[[I1:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH1]] ]
; CHECK-SAME: [ %[[INC1:.+]], %[[LATCH1:.+]] ]
; CHECK-NEXT: detach within %[[SR1]], label %[[BODY1:.+]], label %[[LATCH1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY1]]:
; CHECK-NEXT: %[[RES:.+]] = addrspacecast ptr addrspace(67) %[[RES67]] to ptr
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 4
; CHECK-SAME: i32 5
; CHECK-SAME: ptr %[[RES]]
; CHECK-SAME: i64 %[[I1]]
; CHECK-SAME: i32 0
; CHECK-SAME: i32 8
; CHECK-SAME: i64 0
; CHECK-SAME: ptr @sum
; CHECK-NEXT: reattach within %[[SR1]], label %[[LATCH1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH1]]:
; CHECK-NEXT: %[[INC1:.+]] = add i64 %[[I1]], 1
; CHECK-NEXT: %[[CMP1:.+]] = icmp eq i64 %[[INC1]], %[[N1]]
; CHECK-NEXT: br i1 %[[CMP1]], label %[[LOOPEXIT1:.+]], label %[[HEADER1]]
; CHECK-SAME: !llvm.loop ![[LOOP1:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LOOPEXIT1]]:
; CHECK-NEXT: br label %[[SYNC1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC1]]
; CHECK-NEXT: sync within %[[SR1]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
;
; CHECK-DAG: ![[DEPTH:.+]] = !{!"tapir.loop.perfect.depth", i32 1}
; CHECK-DAG: ![[L1:.+]] = !{!"tapir.loop.perfect.level", i32 1}
; CHECK-DAG: ![[PREPARED:.+]] = !{!"tapir.loop.prepared"}
; CHECK-DAG: ![[REDUCTION:.+]] = !{!"tapir.loop.reduction"}
; CHECK-DAG: ![[TARGET:.+]] = !{!"tapir.loop.target", i32 4}

; CHECK-DAG: ![[LOOP1]] = distinct !{![[LOOP1]], ![[TARGET]], ![[DEPTH]], ![[L1]], ![[REDUCTION]], ![[PREPARED]]}

declare void @sum(ptr, i64)

define void @f1(ptr addrspace(67) %res67, i64 %n1) {
entry:
  %sr1 = tail call token @llvm.syncregion.start()
  %cmpn1 = icmp ult i64 0, %n1
  br i1 %cmpn1, label %ph1, label %sync1

ph1:
  br label %header1

header1:
  %i1 = phi i64 [ 0, %ph1 ], [ %inc1, %latch1 ]
  detach within %sr1, label %body1, label %latch1

body1:
  %res = addrspacecast ptr addrspace(67) %res67 to ptr
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 4, i32 5, ptr %res, i64 %i1, i32 0, i32 8, i64 0, ptr @sum)
  reattach within %sr1, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp.i = icmp eq i64 %inc1, %n1
  br i1 %cmp.i, label %loopexit1, label %header1, !llvm.loop !4

loopexit1:
  br label %sync1

sync1:
  sync within %sr1, label %exit

exit:
  ret void
}

!0 = !{!"tapir.loop.target", i32 4}
!1 = !{!"tapir.loop.perfect.depth", i32 1}
!2 = !{!"tapir.loop.perfect.level", i32 1}
!3 = !{!"tapir.loop.reduction"}
!4 = distinct !{!4, !0, !1, !2, !3}
