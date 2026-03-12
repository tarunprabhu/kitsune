; Check that the names of the outlined kernel functions are as expected. This
; contains both mangled and demangled function names. This checks that the names
; are demangled when generating the outlined kernel name.
;
; RUN: opt --tapir=cuda -passes='loop-spawning' %s \
; RUN:     | %kit-mbc -S \
; RUN:     | FileCheck %s
;
; CHECK-DAG: define {{.+}} @__kitcu_loop_scale_0(
; CHECK-DAG: define {{.+}} @__kitcu_loop_xlate_1(
; CHECK-DAG: define {{.+}} @__kitcu_loop_xlate_2(

define void @_Z5scalePffm(ptr %buf, float %factor, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr float, ptr %buf, i64 %i
  %0 = load float, ptr %arrayidx, align 4
  %mul = fmul float %factor, %0
  store float %mul, ptr %arrayidx, align 4
  reattach within %syncreg, label %latch

latch:
  %i.next = add i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, %n
  br i1 %cmp.i, label %sync, label %header, !llvm.loop !0

sync:
  sync within %syncreg, label %exit

exit:
  ret void
}

define void @xlate(ptr %buf, float %dist, i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  %arrayidx = getelementptr float, ptr %buf, i64 %i
  %0 = load float, ptr %arrayidx, align 4
  %add = fadd float %dist, %0
  store float %add, ptr %arrayidx, align 4
  reattach within %syncreg, label %latch

latch:
  %i.next = add i64 %i, 1
  %cmp.i = icmp eq i64 %i.next, %n
  br i1 %cmp.i, label %sync, label %header, !llvm.loop !4

sync:
  sync within %syncreg, label %preheader2

preheader2:
  %syncreg2 = tail call token @llvm.syncregion.start()
  br label %header2

header2:
  %j = phi i64 [ 0, %preheader2 ], [ %j.next, %latch2 ]
  detach within %syncreg2, label %body2, label %latch2

body2:
  %arrayidx2 = getelementptr float, ptr %buf, i64 %i
  %1 = load float, ptr %arrayidx2, align 4
  %add2 = fadd float %dist, %1
  store float %add2, ptr %arrayidx2, align 4
  reattach within %syncreg2, label %latch2

latch2:
  %j.next = add i64 %j, 1
  %cmp.j = icmp eq i64 %j.next, %n
  br i1 %cmp.j, label %sync2, label %header2, !llvm.loop !5

sync2:
  sync within %syncreg2, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 3}
!2 = !{!"tapir.loop.target", i32 2}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = distinct !{!4, !1, !2, !3}
!5 = distinct !{!5, !1, !2, !3}
