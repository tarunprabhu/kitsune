; The header of a tapir loop must terminate in a detach instruction, even if
; the CFG is "safe".
;
; NOTE: This was originally intended to test that a different error was emitted,
; but that check was moved to later in the code, so the error about the task is
; emitted instead. We leave this test in anyway since the name of the test
; itself acts as an indicator of the various preconditions for tapir loop
; lowering.
;
; RUN: not opt -passes=kit-verify-prelower %s 2>&1 | FileCheck %s
;
; CHECK: cannot get task for tapir loop

define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  br label %detach

detach:
  detach within %syncreg, label %body, label %latch

body:
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !1

exit:
  sync within %syncreg, label %end

end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}
