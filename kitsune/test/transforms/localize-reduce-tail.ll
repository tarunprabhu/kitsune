; When localizing reductions, if the call has been marked as tail, the attribute
; should be removed.
;
; RUN: opt -passes='kit-reduce-localize' -S %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK: {{^[ ]+}}call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 512
; CHECK-SAME: i32 7
; CHECK-SAME: ptr %{{[^,]+}}
; CHECK-SAME: i64 %{{[^,]+}}
; CHECK-SAME: i32 1
; CHECK-SAME: i32 8
; CHECK-SAME: i64 1
; CHECK-SAME: ptr @mul
; CHECK: {{^[ ]+}}call {{.+}} @llvm.kit.reduce
; CHECK-SAME: i32 512
; CHECK-SAME: i32 7
; CHECK-SAME: ptr %{{[^,]+}}
; CHECK-SAME: i64 %{{[^,]+}}
; CHECK-SAME: i32 0
; CHECK-SAME: i32 8
; CHECK-SAME: i64 1
; CHECK-SAME: ptr @mul
define void @f(ptr %res, i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header1

header1:
  %i1 = phi i64 [ 0, %entry ], [ %inc1, %latch1 ]
  detach within %sr, label %body1, label %latch1

body1:
  tail call void(i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 512, i32 7, ptr %res, i64 %i1, i32 0, i32 8, i64 1, ptr @mul)
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
