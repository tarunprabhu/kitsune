; Check all secondary induction variables in a tapir loop are eliminated
; correctly.
;
; RUN: opt -passes=kit-ive -S %s | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-NEXT: [[PH:.+]]:
; CHECK-NEXT: %[[SYNCREG:.+]] = tail call token
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK: %[[PRIMIV:.+]] = phi i64
; CHECK-SAME: [ 0, %[[PH]] ]
; CHECK-SAME: [ %[[NEXT_PRIMIV:.+]], %[[LATCH:.+]] ]
; CHECK: detach within %[[SYNCREG]], label %[[BODY:.+]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: %[[CSTF:.+]] = sitofp i64 %[[PRIMIV]] to float
; CHECK-NEXT: %[[STRIDEF:.+]] = fmul float %[[CSTF]], 2.000000e+00
; CHECK-NEXT: %[[REPLF:.+]] = fadd float 3.000000e+00, %[[STRIDEF]]
; CHECK-NEXT: %[[CSTI:.+]] = trunc i64 %[[PRIMIV]] to i32
; CHECK-NEXT: %[[STRIDEI:.+]] = mul i32 %[[CSTI]], 5
; CHECK-NEXT: %[[REPLI:.+]] = add i32 99, %[[STRIDEI]]
; CHECK-NEXT: call void @ext(i32 %[[REPLI]], float %[[REPLF]])
; CHECK-NEXT: reattach within %[[SYNCREG]], label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[NEXT_PRIMIV]] = add i64 %[[PRIMIV]], 1
; CHECK-NEXT: %[[CMP_PRIMIV:.+]] = icmp eq i64 %[[PRIMIV]]
; CHECK-NEXT: br i1 %[[CMP_PRIMIV]], label %[[EXIT:.+]], label %[[HEADER]]

define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  %j = phi i32 [ 99, %entry ], [ %next.j, %latch ]
  %k = phi float [ 3.0, %entry ], [ %next.k, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  call void @ext(i32 %j, float %k)
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %next.j = add i32 %j, 5
  %next.k = fadd float %k, 2.000000e+00
  %cmp.i = icmp eq i64 %i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  ret void
}

declare void @ext(i32, float)

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}
