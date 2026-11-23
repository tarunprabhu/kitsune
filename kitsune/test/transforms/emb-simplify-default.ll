; By default. the emb-simplify pass should only run on kernel functions.
;
; RUN: %kit-enc %s \
; RUN:     | opt -passes='emb-simplify' \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s

; CHECK-LABEL: @kernel
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[IDX:.+]] = call i64 @index()
; CHECK-NEXT: %[[CMPIDX:.+]] = icmp ult i64 %[[IDX]]
; CHECK-NEXT: br i1 %[[CMPIDX]], label %[[HEADER:.+]], label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: call void @ext(i64 %[[IDX]])
; CHECK-NEXT: br label %[[EXIT]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
define void @kernel(i64 %zero.x, i64 %tc.x) !kit.func !0 {
entry:
  %idx = call i64 @index()
  %cmpidx = icmp ult i64 %idx, %tc.x
  br i1 %cmpidx, label %ph, label %exit

ph:
  %end = add i64 %idx, 1
  br label %header

header:
  %i = phi i64 [ %idx, %ph ], [ %inc, %header ]
  call void @ext(i64 %i)
  %inc = add i64 %idx, 1
  %cmp = icmp eq i64 %inc, %end
  br i1 %cmp, label %exit, label %header, !llvm.loop !1

exit:
  ret void
}

; CHECK-LABEL: @device
; CHECK-NEXT: br label %[[EXIT:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: ret void
define void @device() !kit.func !2 {
  br label %exit

exit:
  ret void
}

declare void @ext(i64)
declare i64 @index()

!0 = distinct !{!0, !3}
!1 = distinct !{!1}
!2 = distinct !{!2, !4}
!3 = !{!"kit.func.kernel", i32 1}
!4 = !{!"kit.func.device"}
