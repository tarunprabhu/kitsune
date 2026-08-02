; Check that the correct kind of instrumentation is added around a tapir loop.
;
; RUN: opt -passes=kit-instrument -S %s \
; RUN:     --kit-instr=generic --kit-instr-only=fg \
; RUN:     | FileCheck %s --check-prefix=GENERIC
;
; RUN: opt -passes=kit-instrument -S %s \
; RUN:     --kit-instr=papi --kit-instr-only=fp --kit-instr-papi=br,lst_ins \
; RUN:     | FileCheck %s --check-prefix=PAPI
;
; RUN: opt -passes=kit-instrument -S %s \
; RUN:     --kit-instr=timer --kit-instr-only=ft \
; RUN:     | FileCheck %s --check-prefix=TIMER
;
; GENERIC: @[[GENERIC:.+]] = private{{.*}} constant [3 x i8] c"fg\00"
; PAPI-DAG: @[[PAPI:.+]] = private{{.*}} constant [3 x i8] c"fp\00"
; PAPI-DAG: @[[BR:.+]] = private{{.*}} constant [3 x i8] c"br\00"
; PAPI-DAG: @[[LST_INS:.+]] = private{{.*}} constant [8 x i8] c"lst_ins\00"
; TIMER: @[[TIMER:.+]] = private{{.*}} constant [3 x i8] c"ft\00"
;
; GENERIC-LABEL: @generic
; GENERIC: br label %[[INSTR_START:.+]]
; GENERIC-EMPTY:
; GENERIC-NEXT: [[INSTR_START]]:
; GENERIC-NEXT: %[[EPOCH:.+]] = call ptr @__kit_instr_start(ptr @[[GENERIC]], i64 0)
; GENERIC-NEXT: br {{.+}} label %[[END:.+]], label %[[PH:.+]]
; GENERIC-EMPTY:
; GENERIC-NEXT: [[PH]]:
; GENERIC: [[END]]:
; GENERIC-NEXT: sync within {{.+}}, label %[[INSTR_STOP:.+]]
; GENERIC-EMPTY:
; GENERIC-NEXT: [[INSTR_STOP]]:
; GENERIC-NEXT: call void @__kit_instr_stop(ptr %[[EPOCH]])
; GENERIC-NEXT: br label %[[RET:.+]]
; GENERIC-EMPTY:
; GENERIC-NEXT: [[RET]]:
; GENERIC-NEXT: ret void
define void @generic(i64 %n) {
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

; PAPI-LABEL: @papi
; PAPI: br label %[[INSTR_START:.+]]
; PAPI-EMPTY:
; PAPI-NEXT: [[INSTR_START]]:
; PAPI-NEXT: %[[EPOCH:.+]] = call ptr (ptr, i64, i32, ...) @__kitpapi_start(ptr @[[PAPI]], i64 0, i32 2, ptr @[[BR]], ptr @[[LST_INS]])
; PAPI-NEXT: br label %[[HEADER:.+]]
; PAPI-EMPTY:
; PAPI-NEXT: [[HEADER]]:
; PAPI: [[BODY:.+]]:
; PAPI: [[LATCH:.+]]:
; PAPI: [[EXIT:.+]]:
; PAPI-NEXT: sync within {{.+}}, label %[[INSTR_STOP:.+]]
; PAPI-EMPTY:
; PAPI-NEXT: [[INSTR_STOP]]:
; PAPI-NEXT: call void @__kitpapi_stop(ptr %[[EPOCH]])
; PAPI-NEXT: br label %[[RET:.+]]
; PAPI-EMPTY:
; PAPI-NEXT: [[RET]]:
; PAPI-NEXT: ret void
define void @papi(i64 %n) {
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
  br i1 %cmp, label %exit, label %header, !llvm.loop !3

exit:
  sync within %sr, label %end

end:
  ret void
}

; TIMER-LABEL: @timer
; TIMER: br label %[[INSTR_START:.+]]
; TIMER-EMPTY:
; TIMER-NEXT: [[INSTR_START]]:
; TIMER-NEXT: %[[EPOCH:.+]] = call ptr @__kittimer_start(ptr @[[TIMER]], i64 0)
; TIMER-NEXT: br label %[[HEADER:.+]]
; TIMER-EMPTY:
; TIMER-NEXT: [[HEADER]]:
; TIMER: [[BODY:.+]]:
; TIMER: [[LATCH:.+]]:
; TIMER: [[EXIT:.+]]:
; TIMER-NEXT: sync within {{.+}}, label %[[INSTR_STOP:.+]]
; TIMER-EMPTY:
; TIMER-NEXT: [[INSTR_STOP]]:
; TIMER-NEXT: call i64 @__kittimer_stop(ptr %[[EPOCH]])
; TIMER-NEXT: br label %[[RET:.+]]
; TIMER-EMPTY:
; TIMER-NEXT: [[RET]]:
; TIMER-NEXT: ret void
define void @timer(i64 %n) {
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
  br i1 %cmp, label %exit, label %header, !llvm.loop !6

exit:
  sync within %sr, label %end

end:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"tapir.loop.name", !"fg"}
!3 = distinct !{!3, !4, !5}
!4 = !{!"tapir.loop.target", i32 512}
!5 = !{!"tapir.loop.name", !"fp"}
!6 = distinct !{!6, !7, !8}
!7 = !{!"tapir.loop.target", i32 1024}
!8 = !{!"tapir.loop.name", !"ft"}
