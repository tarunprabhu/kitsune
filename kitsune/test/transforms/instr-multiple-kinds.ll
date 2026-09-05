; Check that multiple instrumentation functions are added correctly.
;
; RUN: opt -passes=kit-instrument -S %s \
; RUN:     --kit-instr=timer,generic \
; RUN:     | FileCheck %s
;
; CHECK: @[[F:.+]] = private{{.*}} constant [2 x i8] c"f\00"
;
; CHECK-LABEL: @f
; CHECK: br label %[[INSTR_START:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[INSTR_START]]:
; CHECK-NEXT: %[[EPOCHG:.+]] = call ptr @__kit_instr_start(ptr @[[F]], i64 0)
; CHECK-NEXT: %[[EPOCHT:.+]] = call ptr @__kittimer_start(ptr @[[F]], i64 0)
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK: [[BODY:.+]]:
; CHECK: [[LATCH:.+]]:
; CHECK: [[INSTR_STOP:.+]]:
; CHECK-NEXT: call i64 @__kittimer_stop(ptr %[[EPOCHT]])
; CHECK-NEXT: call void @__kit_instr_stop(ptr %[[EPOCHG]])
; CHECK-NEXT: br label %[[SYNC:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[SYNC]]:
; CHECK-NEXT: sync within {{.+}}, label %[[RET:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[RET]]:
; CHECK-NEXT: ret void
define void @f(i64 %n) {
entry:
  %sr = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  detach within %sr, label %body, label %latch

body:
  reattach within %sr, label %latch

latch:
  %next = add i64 %i, 1
  %cmp = icmp eq i64 %next, %n
  br i1 %cmp, label %exit, label %header, !llvm.loop !0

exit:
  sync within %sr, label %end

end:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"tapir.loop.name", !"f"}
