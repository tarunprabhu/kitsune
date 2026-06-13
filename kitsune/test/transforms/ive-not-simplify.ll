; The secondary induction variable elimination pass requires tapir loops to be
; in simplify form before the pass is run. There are several reasons why a loop
; may not be in simplify form, but there is little value in testing each of
; these since that would effectively be testing that Loop::isLoopSimplifyForm()
; is implemented correctly. The purpose of this test is to check that the case
; when isLoopSimplifyForm() returns false is handled correctly.
;
; RUN: not opt -passes='kit-ive' -S %s 2>&1 | FileCheck %s
;
; When running a loop pass, some loop canonicalization passes are always run.
; This includes loop-simplify, so by the time kit-ive sees this, the loop will
; have been simplified. As a result, this test will always fail.
; XFAIL: *
;
; FIXME: We probably should just remove the test since it is unlikely that it
; will ever pass, but having it is at least an indicator that the kit-ive pass
; requires the loop to be in simplify form.
;
; CHECK: loop is not in loop-simplify form

; The loop in this function is not in simplify form because the unique loop
; exit block is not dominated by the loop header.
define void @f0(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp.n = icmp eq i64 %n, 0
  br i1 %cmp.n, label %exit, label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  %j = phi i64 [ 1, %entry ], [ %inc.j, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %inc.j = add i64 %j, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %exit, label %header, !llvm.loop !0

exit:
  sync within %syncreg, label %end

end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"tapir.loop.target", i32 1024}
