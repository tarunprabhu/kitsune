; The loop name attribute cannot be an empty string.
;
; RUN: not llvm-as -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: attribute 'tapir.loop.name': invalid value
; CHECK-SAME: Cannot be an empty string

define void @f(i64 %n) {
entry:
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i ]
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i, !llvm.loop !1

for.i.exit:
  ret void
}

!0 = !{!"tapir.loop.name", !""}
!1 = distinct !{!1, !0}
