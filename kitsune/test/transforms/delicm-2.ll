; Check that the kit-delicm pass sinks instructions into loop nests of depth
; 2 correctly.
;
; RUN: opt -passes='kit-delicm' -S %s 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @p
; CHECK-NEXT: entry:
; CHECK-NEXT: call token @llvm.syncregion.start()
; CHECK-NEXT: br label %for.i.header
; CHECK-EMPTY:
; CHECK-NEXT: for.i.header:
; CHECK-NEXT: %i = phi i64
; CHECK-NEXT: detach
; CHECK-EMPTY:
; CHECK-NEXT: for.i.body:
; CHECK-NEXT: call token @llvm.syncregion.start()
; CHECK-NEXT: br label %for.j.header
; CHECK-EMPTY:
; CHECK-NEXT: for.j.header:
; CHECK-NEXT: %j = phi i64
; CHECK-NEXT: detach
; CHECK-EMPTY:
; CHECK-NEXT: for.j.body:
; CHECK-NEXT: %in = mul i64 %n, %i
; CHECK-NEXT: %in_j = add i64 %in, %j
; CHECK-NEXT: call void @ext(i64 %in_j)
;
define void @p(i64 %m, i64 %n) {
entry:
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  detach within %syncreg.i, label %for.i.body, label %for.i.latch

for.i.body:
  %syncreg.j = tail call token @llvm.syncregion.start()
  %in = mul i64 %n, %i
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.i.body ], [ %j.inc, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  %in_j = add i64 %in, %j
  tail call void @ext(i64 %in_j)
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %j.inc = add i64 %j, 1
  %j.cmp = icmp eq i64 %j.inc, %n
  br i1 %j.cmp, label %for.j.exit, label %for.j.header, !llvm.loop !1

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %i.inc = add i64 %i, 1
  %i.cmp = icmp eq i64 %i, %m
  br i1 %i.cmp, label %for.i.exit, label %for.i.header, !llvm.loop !0

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

declare void @ext(i64)

!0 = distinct !{!0, !2}
!1 = distinct !{!1, !2}
!2 = !{!"tapir.loop.target", i32 2}
