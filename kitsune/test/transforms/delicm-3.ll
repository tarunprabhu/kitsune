; Check that the kit-delicm pass sinks instructions into loop nests of depth
; 2 correctly.
;
; RUN: opt -passes='kit-delicm' -S %s 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: i64 %[[M:[^,]+]],
; CHECK-SAME: i64 %[[N:[^,]+]],
; CHECK-SAME: i64 %[[P:[^)]+]])
; CHECK: %[[SYNCREG_I:.+]] = tail call token @llvm.syncregion.start()
; CHECK: header.i:
; CHECK-NEXT: %[[I:.+]] = phi i64
; CHECK-NEXT: detach within %[[SYNCREG_I]]
; CHECK-EMPTY:
; CHECK-NEXT: body.i:
; CHECK-NEXT: %[[SYNCREG_J:.+]] = tail call token @llvm.syncregion.start()
; CHECK-NEXT: br
; CHECK-EMPTY:
; CHECK-NEXT: ph.j:
; CHECK-NEXT: br label %header.j
; CHECK-EMPTY:
; CHECK-NEXT: header.j:
; CHECK-NEXT: %[[J:.+]] = phi i64
; CHECK-NEXT: detach within %[[SYNCREG_J]]
; CHECK-EMPTY:
; CHECK-NEXT: body.j:
; CHECK-NEXT: %[[SYNCREG_K:.+]] = tail call token @llvm.syncregion.start()
; CHECK-NEXT: br i1
; CHECK-EMPTY:
; CHECK-NEXT: ph.k:
; CHECK-NEXT: br label %header.k
; CHECK-EMPTY:
; CHECK-NEXT: header.k:
; CHECK-NEXT: %[[K:.+]] = phi i64
; CHECK-NEXT: detach within %[[SYNCREG_K]]
; CHECK-EMPTY:
; CHECK-NEXT: body.k:
; CHECK-NEXT: %[[IM:.+]] = mul i64 %[[I]], %m
; CHECK-NEXT: %[[IM_J:.+]] = add i64 %j, %[[IM]]
; CHECK-NEXT: %[[IMN_JN:.+]] = mul i64 %[[IM_J]], %n
; CHECK-NEXT: %[[IMN_JN_K:.+]] = add i64 %[[K]], %[[IMN_JN]]
; CHECK-NEXT: call void @ext(i64 %[[IMN_JN_K]])
; CHECK-NEXT: reattach within %[[SYNCREG_K]]

define void @f(i64 %m, i64 %n, i64 %p) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  %cmp.m = icmp sgt i64 %m, 0
  br i1 %cmp.m, label %ph.i, label %sync.i

ph.i:
  %cmp.n = icmp sgt i64 %n, 0
  %cmp.p = icmp sgt i64 %p, 0
  br label %header.i

header.i:
  %i = phi i64 [ 0, %ph.i ], [ %inc.i, %latch.i ]
  detach within %syncreg.i, label %body.i, label %latch.i

body.i:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br i1 %cmp.n, label %ph.j, label %sync.j

ph.j:
  %im = mul i64 %i, %m
  br label %header.j

header.j:
  %j = phi i64 [ 0, %ph.j ], [ %inc.j, %latch.j ]
  detach within %syncreg.j, label %body.j, label %latch.j

body.j:
  %syncreg.k = tail call token @llvm.syncregion.start()
  br i1 %cmp.p, label %ph.k, label %sync.k

ph.k:
  %im_j = add i64 %j, %im
  %imn_jn = mul i64 %im_j, %n
  br label %header.k

header.k:
  %k = phi i64 [ 0, %ph.k ], [ %inc.k, %latch.k ]
  detach within %syncreg.k, label %body.k, label %latch.k

body.k:
  %imn_jn_k = add i64 %k, %imn_jn
  call void @ext(i64 %imn_jn_k)
  reattach within %syncreg.k, label %latch.k

latch.k:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %exit.k, label %header.k, !llvm.loop !2

exit.k:
  br label %sync.k

sync.k:
  sync within %syncreg.k, label %end.k

end.k:
  reattach within %syncreg.j, label %latch.j

latch.j:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %exit.j, label %header.j, !llvm.loop !1

exit.j:
  br label %sync.j

sync.j:
  sync within %syncreg.j, label %end.j

end.j:
  reattach within %syncreg.i, label %latch.i

latch.i:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %exit.i, label %header.i, !llvm.loop !0

exit.i:
  br label %sync.i

sync.i:
  sync within %syncreg.i, label %end.i

end.i:
  ret void
}

declare void @ext(i64)

!0 = distinct !{!0, !3}
!1 = distinct !{!1, !3}
!2 = distinct !{!2, !3}
!3 = !{!"tapir.loop.target", i32 2}
