; Check that generic instrumentation is correctly added around a tapir loop.
;
; RUN: opt -passes="kit-instrument" -S %s \
; RUN:     --kit-instr-unit=thread  --kit-instr=generic \
; RUN:     | FileCheck %s
;
; CHECK: @[[NAME:.+]] = private{{.*}} constant [9 x i8] c"pthreads\00"
;
; CHECK-LABEL: @f
; CHECK: br label %[[HEADER:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[HEADER]]:
; CHECK: detach within {{.+}}, label %[[INSTR_START:.+]], label %[[LATCH:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[INSTR_START]]:
; CHECK-NEXT: %[[THRD:.+]] = call i64 @llvm.kit.cpu.thread.id(i32 1024)
; CHECK-NEXT: %[[EPOCH:.+]] = call ptr @__kit_instr_start(ptr @[[NAME]], i64 %[[THRD]])
; CHECK-NEXT: br label %[[BODY:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[BODY]]:
; CHECK-NEXT: call void @ext
; CHECK-NEXT: br label %[[INSTR_STOP:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[INSTR_STOP]]:
; CHECK-NEXT: call void @__kit_instr_stop(ptr %[[EPOCH]])
; CHECK-NEXT: reattach within {{.+}}, label %[[LATCH]]
; CHECK-EMPTY:
; CHECK-NEXT: [[LATCH]]:
; CHECK: br {{.+}}, label %{{.+}}, label %[[HEADER]]

declare void @ext(i64)

define void @f(i64 %n) {
entry:
  %sr = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  detach within %sr, label %body, label %latch

body:
  call void @ext(i64 %i)
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
!1 = !{!"tapir.loop.target", i32 1024}
!2 = !{!"tapir.loop.name", !"pthreads"}
