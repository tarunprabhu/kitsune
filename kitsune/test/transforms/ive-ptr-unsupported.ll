; Only certain forms of pointer induction are currently supported. This is not
; one of them.
;
; RUN: not opt -passes='kit-ive' -S %s 2>&1 | FileCheck %s
;
; CHECK: secondary induction variable has unsupported pointer induction

define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  %j = phi ptr [ null, %entry ], [ %next.j, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %next.j = load ptr, ptr %j
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  sync within %syncreg, label %end

end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}
