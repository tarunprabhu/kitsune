; If the --funcs option is given, check that it works as expected.
;
; RUN: %kit-sort --funcs=f,sorted %s | FileCheck %s

; CHECK-LABEL: @ppp
; CHECK: entry:
; CHECK: for.i.body:
; CHECK: for.i.header:
; CHECK: for.j.header:
; CHECK: for.j.latch:
; CHECK: for.j.exit:
; CHECK: for.i.exit:
; CHECK: for.j.end:
; CHECK: for.k.body:
; CHECK: for.j.body:
; CHECK: for.k.header:
; CHECK: for.k.latch:
; CHECK: for.k.end:
; CHECK: for.i.latch:
; CHECK: for.k.exit:
define void @ppp(i64 %tc.z, i64 %tc.y, i64 %tc.x, i1 %cond) {
entry:
  br i1 %cond, label %for.i.header, label %for.i.exit

for.i.body:
  br label %for.j.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  br label %for.i.body

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %inc.j, %for.j.latch ]
  br label %for.j.body

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %tc.y
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !1

for.j.exit:
  br label %for.j.end

for.i.exit:
  ret void

for.j.end:
  br label %for.i.latch

for.k.body:
  br label %for.k.latch

for.j.body:
  br label %for.k.header

for.k.header:
  %k = phi i64 [ 0, %for.j.body ], [ %inc.k, %for.k.latch ]
  br label %for.k.body

for.k.latch:
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %tc.x
  br i1 %cmp.k, label %for.k.exit, label %for.k.header, !llvm.loop !2

for.k.end:
  br label %for.j.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %tc.z
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.k.exit:
  br label %for.k.end
}

; CHECK-LABEL: @f
; CHECK: entry:
; CHECK: check:
; CHECK: if.false:
; CHECK: if.true:
; CHECK: exit:
define void @f(i1 %cond) {
entry:
  br label %check

exit:
  ret void

check:
  br i1 %cond, label %if.true, label %if.false

if.true:
  br label %exit

if.false:
  br label %exit
}

; CHECK-LABEL: @sorted
; CHECK: entry:
; CHECK: body:
; CHECK: end:
define void @sorted() {
entry:
  br label %body

body:
  br label %end

end:
  ret void
}

!0 = distinct !{!0}
!1 = distinct !{!1}
!2 = distinct !{!2}
