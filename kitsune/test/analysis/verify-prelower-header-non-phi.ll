; Non-PHI instructions are not allowed in the tapir loop header.
;
; RUN: not opt --tapir=nolo -passes=kit-verify-prelower %s 2>&1 | FileCheck %s
;
; CHECK: tapir loop header must contain only phi nodes

define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  %i.trunc = trunc i64 %i to i32
  detach within %syncreg, label %body, label %latch

body:
  call void @ext(i32 %i.trunc)
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  sync within %syncreg, label %end

end:
  ret void
}

declare void @ext(i32)

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}
