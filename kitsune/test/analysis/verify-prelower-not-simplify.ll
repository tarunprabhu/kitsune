; Since the kit-verify-prelower pass is run just before loop spawning, all tapir
; loops must be in loop-simplify form. This is generally required by nearly all
; passes that operate on tapir loops, so we test for it several times. The
;
; RUN: not opt -passes='kit-verify-prelower' %s -disable-output 2>&1 \
; RUN:     | FileCheck %s
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
