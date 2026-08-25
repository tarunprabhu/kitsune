; In some cases, the updated value of the induction variable is not used in the
; termination condition of the loop latch. This should not result in an error.
; This is a reduced example from actual code, where `%tc = add i64 %n, -2` was
; present.
;
; RUN: opt -passes=kit-verify-prelower -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{^.+$}}

define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %tc = add i64 %n, -2
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %next.i, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  reattach within %syncreg, label %latch

latch:
  %next.i = add nuw i64 %i, 1
  %cmp.i = icmp eq i64 %i, %tc
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  sync within %syncreg, label %end

end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}
