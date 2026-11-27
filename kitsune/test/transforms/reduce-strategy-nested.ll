; If reductions are nested within a serial loop within a tapir loop, an
; appropriate strategy should be computed for them too.
;
; RUN: opt -passes='kit-reduce-strategy' -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[TOTAL:.[^,]+]]
; CHECK-SAME: ptr %[[PARTS:[^,]+]]
; CHECK-SAME: i64 %[[N:[^)]+]]
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SR:.+]] = call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[I:.+]] = phi i64
; CHECK-SAME: [ 0, %[[ENTRY]] ]
; CHECK-SAME: [ %[[INCI:.+]], %[[LATCH:.+]] ]
; CHECK: detach within %[[SR]], label %[[BODY:.+]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: br label %[[NESTED:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[NESTED]]:
; CHECK-NEXT: %[[J:.+]] = phi i64
; CHECK-SAME: [ 0, %[[BODY]] ]
; CHECK-SAME: [ %[[INCJ:.+]], %[[NESTED]] ]
; CHECK-NEXT: and
; CHECK-NEXT: %[[DEST:.+]] = getelementptr i64, ptr %[[PARTS]], i64 %{{.+}}
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 2
; CHECK-SAME: i32 5
; CHECK-SAME: ptr %[[DEST]]
; CHECK-SAME: i64 %[[I]]
; CHECK-SAME: i32 2
; CHECK-NEXT: %[[INCJ]] = add i64 %[[J]], 1
; CHECK-NEXT: %[[CMPJ:.+]] = icmp eq i64 %[[INCJ]], %[[N]]
; CHECK-NEXT: br i1 %[[CMPJ]], label %[[REATTACH:.+]], label %[[NESTED]]
; CHECK-SAME: !llvm.loop ![[LOOP_N:[0-9]]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH]]:
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 2
; CHECK-SAME: i32 5
; CHECK-SAME: ptr %[[TOTAL]]
; CHECK-SAME: i64 %[[I]]
; CHECK-SAME: i32 10
; CHECK-NEXT: reattach within %[[SR]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[INCI:.+]] = add i64 %[[I]], 1
; CHECK-NEXT: %[[CMPI:.+]] = icmp eq i64 %[[INCI]], %[[N]]
; CHECK-NEXT: br i1 %[[CMPI]], label %[[EXIT:.+]], label %[[HEADER]]
; CHECK-SAME: !llvm.loop ![[LOOP:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void

define void @f(ptr %total, ptr %parts, i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  br label %nested

nested:
  %j = phi i64 [ 0, %body ], [ %incj, %nested ]
  %zero.or.one = and i64 %i, 1
  %dest = getelementptr i64, ptr %parts, i64 %zero.or.one
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 2, i32 5, ptr %dest, i64 %i, i32 0, i32 8, i64 0, ptr @sum)
  %incj = add i64 %j, 1
  %cmpj = icmp eq i64 %incj, %n
  br i1 %cmpj, label %reattach, label %nested, !llvm.loop !3

reattach:
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 2, i32 5, ptr %total, i64 %i, i32 0, i32 8, i64 0, ptr @sum)
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %exit, label %header, !llvm.loop !2

exit:
  ret void
}

declare void @sum(ptr, i64)

!0 = !{!"tapir.loop.target", i32 2}
!1 = !{!"tapir.loop.reduction"}
!2 = distinct !{!2, !0, !1}
!3 = distinct !{!3}
