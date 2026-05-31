; If any unreachable blocks are present, they must not be removed from the
; result.
;
; RUN: %kit-sort %s | FileCheck %s

; CHECK-LABEL: @f
; CHECK: entry:
; CHECK: header.i:
; CHECK: body.i:
; CHECK: header.j1:
; CHECK: body.j1:
; CHECK: latch.j1:
; CHECK: exit.j1:
; CHECK: latch.i
; CHECK: end:
; CHECK: header.j2:
; CHECK: latch.j2:
; CHECK: body.j2:
define void @f(i64 %m, i64 %n) {
entry:
  br label %header.i

header.i:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch.i ]
  br label %body.i

end:
  ret void

body.i:
  br label %header.j1

header.j1:
  %j1 = phi i64 [ 0, %header.i ], [ %inc.j1, %latch.j1 ]
  br label %body.j1

exit.j1:
  br label %latch.i

body.j1:
  br label %latch.j1

header.j2: ; only reachable from the loop latch, but not from anywhere else
  %j2 = phi i64 [ 0, %exit.j1 ], [ %inc.j2, %latch.j2 ]
  br label %body.j2

latch.j1:
  %inc.j1 = add i64 %j1, 1
  %cmp.j1 = icmp eq i64 %inc.j1, %n
  br i1 %cmp.j1, label %exit.j1, label %header.j1, !llvm.loop !1

latch.i:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %end, label %header.i, !llvm.loop !0

latch.j2:
  %inc.j2 = add i64 %j2, 1
  %cmp.j2 = icmp eq i64 %inc.j2, %n
  br i1 %cmp.j2, label %latch.i, label %header.j2, !llvm.loop !2

body.j2:
  br label %latch.j2
}

!0 = distinct !{!0}
!1 = distinct !{!1}
!2 = distinct !{!2}
