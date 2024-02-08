; Simple test of the kit-delicm pass when there is only one loop.
;
; NOTE: As of Apr-2026, this will have no effect since the pass requires the
; loop to have depth 2 or greater. At some point, this may change and we may
; support this optimization on loops of depth 1 as well.
;
; RUN: opt -passes='kit-delicm' -S %s 2>&1 \
; RUN:     | FileCheck %s

; CHECK-LABEL: @s
; CHECK-NEXT: entry:
; CHECK-NEXT: %t = mul i64 %n, 4
; CHECK-NEXT: br label %for.i
; CHECK-EMPTY:
; CHECK-NEXT: for.i:
; CHECK-NEXT: phi i64
; CHECK-NEXT: call void @ext(i64 %t)
define void @s(i64 %n) {
entry:
  %t = mul i64 %n, 4
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.i ]
  call void @ext(i64 %t)
  %i.inc = add i64 %i, 1
  %i.cmp = icmp eq i64 %i, %n
  br i1 %i.cmp, label %exit, label %for.i, !llvm.loop !1

exit:
  ret void
}

; CHECK-LABEL: @p
; CHECK-NEXT: entry:
; CHECK-NEXT: call token @llvm.syncregion.start()
; CHECK-NEXT: %t = mul i64 %n, 4
; CHECK-NEXT: br label %for.i.header
; CHECK-EMPTY:
; CHECK-NEXT: for.i.header:
; CHECK-NEXT: phi i64
; CHECK-NEXT: detach
; CHECK-EMPTY:
; CHECK-NEXT: for.i.body:
; CHECK-NEXT: call void @ext(i64 %t)
define void @p(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %t = mul i64 %n, 4
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.i.latch ]
  detach within %syncreg, label %for.i.body, label %for.i.latch

for.i.body:
  call void @ext(i64 %t)
  reattach within %syncreg, label %for.i.latch

for.i.latch:
  %i.inc = add i64 %i, 1
  %i.cmp = icmp eq i64 %i, %n
  br i1 %i.cmp, label %for.i.exit, label %for.i.header, !llvm.loop !2

for.i.exit:
  sync within %syncreg, label %exit

exit:
  ret void
}

declare void @ext(i64)

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1}
!2 = distinct !{!2, !0}
