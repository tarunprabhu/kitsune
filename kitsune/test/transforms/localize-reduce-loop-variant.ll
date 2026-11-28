; If the destination of a reduction is loop-variant, it should not be localized.
;
; RUN: opt -passes='kit-reduce-localize' -S %s | FileCheck %s

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
; CHECK-SAME: [ %[[INC:.+]], %[[LATCH:.+]] ]
; CHECK: detach within %[[SR]], label %[[BODY:.+]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: %[[LOCAL:.+]] = alloca i64
; CHECK-NEXT: store i64 0, ptr %[[LOCAL]]
; CHECK-NEXT: and
; CHECK-NEXT: %[[DEST:.+]] = getelementptr i64, ptr %[[PARTS]], i64 %{{.+}}
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 512
; CHECK-SAME: i32 5
; CHECK-SAME: ptr %[[DEST]]
; CHECK-SAME: i64 %[[I]]
; CHECK-SAME: i32 0
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 512
; CHECK-SAME: 5
; CHECK-SAME: ptr %[[LOCAL]]
; CHECK-SAME: i64 %[[I]]
; CHECK-SAME: i32 1
; CHECK-NEXT: br label %[[REATTACH:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[REATTACH]]:
; CHECK-NEXT: %[[V:.+]] = load i64
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 512
; CHECK-SAME: i32 5
; CHECK-SAME: ptr %[[TOTAL]]
; CHECK-SAME: i64 %[[V]]
; CHECK-SAME: i32 0
; CHECK-NEXT: reattach within %[[SR]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[INC:.+]] = add i64 %[[I]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp eq i64 %[[INC]], %[[N]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]]
; CHECK-SAME: !llvm.loop ![[LOOP:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void

; CHECK-DAG: ![[LOCALIZED:[0-9]+]] = !{!"tapir.loop.localized"}
; CHECK-DAG: ![[LOOP]] = distinct !{![[LOOP]], {{.*}} ![[LOCALIZED]]

define void @f(ptr %total, ptr %parts, i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  %zero.or.one = and i64 %i, 1
  %dest = getelementptr i64, ptr %parts, i64 %zero.or.one
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 512, i32 5, ptr %dest, i64 %i, i32 0, i32 8, i64 0, ptr @sum)
  call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 512, i32 5, ptr %total, i64 %i, i32 0, i32 8, i64 0, ptr @sum)
  br label %reattach

reattach:
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %exit, label %header, !llvm.loop !5

exit:
  ret void
}

declare void @sum(ptr, i64)

!0 = !{!"tapir.loop.target", i32 512}
!1 = !{!"tapir.loop.perfect.level", i32 1}
!2 = !{!"tapir.loop.perfect.depth", i32 1}
!3 = !{!"tapir.loop.reduction"}
!4 = !{!"tapir.loop.prepared"}
!5 = distinct !{!5, !0, !1, !2, !3, !4}
