; Check that the kit-verify-prelower pass does not produce any output when
; there are no tapir loops in the module.
;
; RUN: opt --tapir=serial -passes='kit-verify-prelower' %s 2>&1 \
; RUN:     -disable-output \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{.+}}

target triple = "x86_64-unknown-linux-gnu"

define void @sss(i64 %m, i64 %n, i64 %p) {
entry:
  %cmp.m = icmp sgt i64 %m, 0
  %cmp.n = icmp sgt i64 %n, 0
  %cmp.p = icmp sgt i64 %p, 0
  br i1 %cmp.m, label %for.i.header, label %exit

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  br i1 %cmp.n, label %for.j.header, label %for.i.latch

for.j.header:
  %j = phi i64 [ 0, %for.i.header ], [ %inc.j, %for.j.latch ]
  br i1 %cmp.p, label %for.k.header, label %for.j.latch

for.k.header:
  %k = phi i64 [ 0, %for.j.header ], [ %inc.k, %for.k.header ]
  %inc.k = add nuw nsw i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %p
  br i1 %cmp.k, label %for.j.latch, label %for.k.header, !llvm.loop !0

for.j.latch:
  %inc.j = add nuw nsw i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.i.latch, label %for.j.header, !llvm.loop !1

for.i.latch:
  %inc.i = add nuw nsw i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %m
  br i1 %cmp.i, label %exit, label %for.i.header, !llvm.loop !2

exit:
  ret void
}

!0 = !{!0}
!1 = !{!1}
!2 = !{!2}
