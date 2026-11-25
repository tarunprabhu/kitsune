; If the destination of a reduction is loop-variant, only the Direct strategy
; may be used.
;
; RUN: opt -passes='kit-reduce-strategy' -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK-SAME: ptr %[[TOTAL:.[^,]+]]
; CHECK-SAME: ptr %[[PARTS:[^,]+]]
; CHECK: [[ENTRY:.+]]:
; CHECK: [[HEADER:.+]]:
; CHECK: %[[I:.+]] = phi i64
; CHECK: detach within
; CHECK: %[[DEST:.+]] = getelementptr i64, ptr %[[PARTS]], i64 %{{.+}}
; CHECK: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 512
; CHECK-SAME: i32 5
; CHECK-SAME: ptr %[[DEST]]
; CHECK-SAME: i64 %[[I]]
; CHECK-SAME: i32 2
; CHECK-NEXT: call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 512
; CHECK-SAME: i32 5
; CHECK-SAME: ptr %[[TOTAL]]
; CHECK-SAME: i64 %[[I]]
; CHECK-SAME: i32 2
; CHECK: reattach within
; CHECK: br i1 %{{.+}}, label %{{.+}}, label %[[HEADER]]
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
