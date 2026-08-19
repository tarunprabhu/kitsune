; Reductions in nested tapir loops are not yet supported.
;
; RUN: not opt -passes="kit-verify-early" %s 2>&1 | FileCheck %s

; forall (...)
;   forall (...)
;     kit.reduce.0
;
; CHECK: NOT YET IMPLEMENTED: nested parallel reductions
;
define void @pp(i64 %n) {
entry:
  %result = alloca i64
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.j.ph, label %for.i.latch

for.j.ph:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.j.ph ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  call void (i32, i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 1, i32 1, ptr %result, i32 8, i64 %j, i64 0, ptr null)
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !2

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !1

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

; forall (...)
;   forall (...)
;     for (...)
;       kit.reduce.0
;
; CHECK: NOT YET IMPLEMENTED: nested parallel reductions
;
define void @pps(i64 %n) {
entry:
  %result = alloca i64
  %syncreg.i = tail call token @llvm.syncregion.start()
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %for.i.latch ]
  detach within %syncreg.i, label %for.j.ph, label %for.i.latch

for.j.ph:
  %syncreg.j = tail call token @llvm.syncregion.start()
  br label %for.j.header

for.j.header:
  %j = phi i64 [ 0, %for.j.ph ], [ %inc.j, %for.j.latch ]
  detach within %syncreg.j, label %for.j.body, label %for.j.latch

for.j.body:
  br label %for.k

for.k:
  %k = phi i64 [ 0, %for.j.body ], [ %inc.k, %for.k ]
  call void (i32, i32, ptr, i32, i64, i64, ptr, ...) @llvm.kit.reduce.0(i32 1, i32 1, ptr %result, i32 8, i64 %j, i64 0, ptr null)
  %inc.k = add i64 %k, 1
  %cmp.k = icmp eq i64 %inc.k, %n
  br i1 %cmp.k, label %for.k.exit, label %for.k, !llvm.loop !5

for.k.exit:
  br label %for.k.end

for.k.end:
  reattach within %syncreg.j, label %for.j.latch

for.j.latch:
  %inc.j = add i64 %j, 1
  %cmp.j = icmp eq i64 %inc.j, %n
  br i1 %cmp.j, label %for.j.exit, label %for.j.header, !llvm.loop !4

for.j.exit:
  sync within %syncreg.j, label %for.j.end

for.j.end:
  reattach within %syncreg.i, label %for.i.latch

for.i.latch:
  %inc.i = add i64 %i, 1
  %cmp.i = icmp eq i64 %inc.i, %n
  br i1 %cmp.i, label %for.i.exit, label %for.i.header, !llvm.loop !3

for.i.exit:
  sync within %syncreg.i, label %for.i.end

for.i.end:
  ret void
}

!0 = !{!"tapir.loop.target", i32 1}
!1 = distinct !{!1, !0}
!2 = distinct !{!2, !0}
!3 = distinct !{!3, !0}
!4 = distinct !{!4, !0}
!5 = distinct !{!5}
