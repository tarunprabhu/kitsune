; Check that reductions in a loop containing a single reduction with the
; reattach instruction in the same block works as expected.
;
; RUN: opt -passes='kit-reduce-localize' -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-SAME: %[[RES:[^,]+]]
; CHECK-SAME: %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK: %[[SR:.+]] = call token @llvm.syncregion.start()
; CHECK: br label %[[HEADER1:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER1]]
; CHECK-NEXT: %[[I1:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INC1:.+]], %[[LATCH1:.+]] ]
; CHECK-NEXT: detach within %[[SR]], label %[[BODY1:.+]], label %[[LATCH1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY1]]:
; CHECK-NEXT: %[[LOCAL1:.+]] = alloca i64
; CHECK-NEXT: store i64 1, ptr %[[LOCAL1]]
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 512
; CHECK-SAME: i32 7
; CHECK-SAME: ptr %[[LOCAL1]]
; CHECK-SAME: i64 %[[I1]]
; CHECK-SAME: i32 1
; CHECK-SAME: i32 8
; CHECK-SAME: i64 1
; CHECK-SAME: ptr @mul
; CHECK-NEXT: %[[V:.+]] = load i64, ptr %[[LOCAL1]]
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 512
; CHECK-SAME: i32 7
; CHECK-SAME: ptr %[[RES]]
; CHECK-SAME: i64 %[[V]]
; CHECK-SAME: i32 2
; CHECK-SAME: i32 8
; CHECK-SAME: i64 1
; CHECK-SAME: ptr @mul
; CHECK-NEXT: reattach within %[[SR]], label %[[LATCH1]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH1]]:
; CHECK-NEXT: %[[INC1]] = add i64 %[[I1]], 1
; CHECK-NEXT: %[[CMP1:.+]] = icmp eq i64 %[[INC1]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP1]], label %{{.+}}, label %[[HEADER1]]
; CHECK-SAME: !llvm.loop ![[LOOP1:[0-9]+]]

; CHECK-DAG: ![[LOCALIZED:[0-9]+]] = !{!"tapir.loop.localized"}
; CHECK-DAG: ![[LOOP1]] = distinct !{![[LOOP1]], {{.*}} ![[LOCALIZED]]}

define void @f(ptr %res, i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i1 = phi i64 [ 0, %entry ], [ %inc1, %latch1 ]
  detach within %sr, label %body1, label %latch1

body1:
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 512, i32 7, ptr %res, i64 %i1, i32 0, i32 8, i64 1, ptr @mul)
  reattach within %sr, label %latch1

latch1:
  %inc1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %inc1, %n
  br i1 %cmp1, label %sync1, label %header1, !llvm.loop !5

sync1:
  sync within %sr, label %exit

exit:
  ret void
}

declare void @mul(ptr, i64)

!0 = !{!"tapir.loop.target", i32 512}
!1 = !{!"tapir.loop.perfect.level", i32 1}
!2 = !{!"tapir.loop.perfect.depth", i32 1}
!3 = !{!"tapir.loop.reduction"}
!4 = !{!"tapir.loop.prepared"}
!5 = distinct !{!5, !0, !1, !2, !3, !4}
!6 = distinct !{!6}
