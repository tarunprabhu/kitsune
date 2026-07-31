; If --kit-instr-unit=default is provided, check that the default set of
; constructs is instrumented. This test only has tapir loops. If new
; Kitsune-specific constructs are added, those be tested in a separate test.
;
; RUN: opt -passes=kit-instrument -S %s \
; RUN:     --kit-instr=generic --kit-instr-unit=default \
; RUN:     | FileCheck %s
;
; CHECK: @[[NAME1:.+]] = private{{.*}} constant [6 x i8] c"loop1\00"
; CHECK: @[[NAME2:.+]] = private{{.*}} constant [6 x i8] c"loop2\00"
; CHECK: @[[NAME3:.+]] = private{{.*}} constant [6 x i8] c"loop3\00"
;
; CHECK-LABEL: @f1
; CHECK: %[[EPOCH1:.+]] = call ptr @__kit_instr_start(ptr @[[NAME1]], i64 0)
; CHECK: phi
; CHECK: !llvm.loop ![[LOOP1:[0-9]+]]
; CHECK: sync
; CHECK: call void @__kit_instr_stop(ptr %[[EPOCH1]])
; CHECK: ret void
;
; CHECK-LABEL @f2
; CHECK: %[[EPOCH2:.+]] = call ptr @__kit_instr_start(ptr @[[NAME2]], i64 0)
; CHECK: phi
; CHECK: !llvm.loop ![[LOOP2:[0-9]+]]
; CHECK: sync
; CHECK: call void @__kit_instr_stop(ptr %[[EPOCH2]])
; CHECK: %[[EPOCH3:.+]] = call ptr @__kit_instr_start(ptr @[[NAME3]], i64 0)
; CHECK: phi
; CHECK: !llvm.loop ![[LOOP3:[0-9]+]]
; CHECK: sync
; CHECK: call void @__kit_instr_stop(ptr %[[EPOCH3]])
; CHECK: ret void

define void @f1(i64 %n) {
entry:
  %sr = tail call token @llvm.syncregion.start()
  %guard = icmp eq i64 %n, 0
  br i1 %guard, label %end, label %ph

ph:
  br label %header

header:
  %i = phi i64 [ 0, %ph ], [ %next, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %next = add i64 %i, 1
  %cmp = icmp eq i64 %next, %n
  br i1 %cmp, label %exit, label %header, !llvm.loop !0

exit:
  br label %end

end:
  sync within %sr, label %ret

ret:
  ret void
}

define void @f2(i64 %n) {
ph1:
  %sr = tail call token @llvm.syncregion.start()
  br label %header1

header1:
  %i1 = phi i64 [ 0, %ph1 ], [ %next1, %latch1 ]
  detach within %sr, label %body1, label %latch1

body1:
  reattach within %sr, label %latch1

latch1:
  %next1 = add i64 %i1, 1
  %cmp1 = icmp eq i64 %next1, %n
  br i1 %cmp1, label %exit1, label %header1, !llvm.loop !3

exit1:
  sync within %sr, label %ph2

ph2:
  br label %header2

header2:
  %i2 = phi i64 [ 0, %ph2 ], [ %next2, %latch2 ]
  detach within %sr, label %body2, label %latch2

body2:
  reattach within %sr, label %latch2

latch2:
  %next2 = add i64 %i2, 1
  %cmp2 = icmp eq i64 %next2, %n
  br i1 %cmp2, label %exit2, label %header2, !llvm.loop !6

exit2:
  sync within %sr, label %ret

ret:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"tapir.loop.name", !"loop1"}
!3 = distinct !{!3, !4, !5}
!4 = !{!"tapir.loop.target", i32 512}
!5 = !{!"tapir.loop.name", !"loop2"}
!6 = distinct !{!6, !7, !8}
!7 = !{!"tapir.loop.target", i32 1024}
!8 = !{!"tapir.loop.name", !"loop3"}
