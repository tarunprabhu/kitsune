; Check that the output of the loop-spawning pass is as expected. The following
; are expected:
;
;   - The tapir loop has not been outlined
;
;   - Detach and reattach instructions in the loop have been replaced with
;     unconditional branches
;
;   - Sync instructions have not been removed since the loop spawning pass does
;     not invoke the lowerSync callback on the tapir target object
;
;   - All tapir loop attributes as well mandatory LLVM loops attributes added to
;     tapir loops have been removed.
;
; ------------------------------------------------------------------------------
;
; RUN: opt --tapir=serial -passes='loop-spawning' -S %s | FileCheck %s
;
; CHECK-LABEL: @p
; CHECK-NEXT: [[ENTRY:.+]]:
; CHECK-NEXT: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: %[[IV:.+]] = phi i64
; CHECK-NEXT: br label %[[BODY:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: br label %[[LATCH:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK-NEXT: %[[IV_NEXT:.+]] = add i64 %[[IV]], 1
; CHECK-NEXT: %[[CMP:.+]] = icmp {{.+}} %[[IV_NEXT]]
; CHECK-NEXT: br i1 %[[CMP]], label %[[EXIT:.+]], label %[[HEADER]]
; CHECK-SAME: !llvm.loop ![[MD:[0-9]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[EXIT]]:
; CHECK-NEXT: sync within %[[SYNCREG]], label %[[END:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END]]:
; CHECK-NEXT: ret void
;
; CHECK: ![[MD]] = distinct !{![[MD]]}

define void @p(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg, label %for.i.end

for.i.end:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"llvm.loop.unroll.disable"}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.spawn.strategy", i32 1}
