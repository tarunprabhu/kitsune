; Check that --kit-instr-unit=all correctly instruments both loops and threads.
; If any new constructs are added to Kitsune, a separate test should be added
; that checks --kit-instr-unit=all.
;
; RUN: opt -passes=kit-instrument -S %s \
; RUN:     --kit-instr=generic --kit-instr-unit=all \
; RUN:     | FileCheck %s
;
; CHECK: @[[NAME:.+]] = private{{.*}} constant [5 x i8] c"loop\00"
;
; CHECK-LABEL: @f
; CHECK: br label %[[INSTR_START_O:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[INSTR_START_O]]:
; CHECK-NEXT: %[[EPOCHO:.+]] = call ptr @__kit_instr_start(ptr @[[NAME]], i64 0)
; CHECK-NEXT: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK-NEXT: phi i64
; CHECK-NEXT: detach within {{.+}}, label %[[INSTR_START_I:.+]], label %[[LATCH:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[INSTR_START_I]]:
; CHECK-NEXT: %[[THRD:.+]] = call i64 @llvm.kit.cpu.thread.id(i32 512)
; CHECK-NEXT: %[[EPOCHI:.+]] = call ptr @__kit_instr_start(ptr @[[NAME]], i64 %[[THRD]])
; CHECK-NEXT: br label %[[BODY:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: br label %[[INSTR_STOP_I:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[INSTR_STOP_I]]:
; CHECK-NEXT: call void @__kit_instr_stop(ptr %[[EPOCHI]])
; CHECK-NEXT: reattach within {{.+}}, label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK: br {{.+}}, label %[[INSTR_STOP_O:.+]], label %[[HEADER]]
; CHECK-EMPTY:
; CHECK-NEXT: [[INSTR_STOP_O]]:
; CHECK-NEXT: call void @__kit_instr_stop(ptr %[[EPOCHO]])
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
!1 = !{!"tapir.loop.target", i32 512}
!2 = !{!"tapir.loop.name", !"loop"}
