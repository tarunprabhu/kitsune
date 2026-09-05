; Check that multiple names passed to --kit-instr-loop-only works as expected
;
; RUN: opt -passes=kit-instrument -S %s \
; RUN:     --kit-instr=timer --kit-instr-only=loop1,loop3 \
; RUN:     | FileCheck %s
;
; CHECK-DAG: @[[LOOP1:.+]] = private{{.*}} constant [6 x i8] c"loop1\00"
; CHECK-DAG: @[[LOOP3:.+]] = private{{.*}} constant [6 x i8] c"loop3\00"
;
; CHECK: %[[EPOCH1:.+]] = call ptr @__kittimer_start(ptr @[[LOOP1]], i64 0)
; CHECK: phi i64
; CHECK: !llvm.loop ![[LOOP1:[0-9]+]]
; CHECK: call i64 @__kittimer_stop(ptr %[[EPOCH1]])
; CHECK: sync within
; CHECK-NOT: __kittimer_start
; CHECK: phi i64
; CHECK: !llvm.loop ![[LOOP2:[0-9]+]]
; CHECK: sync within
; CHECK-NOT: __kittimer_stop
; CHECK: %[[EPOCH3:.+]] = call ptr @__kittimer_start(ptr @[[LOOP3]], i64 0)
; CHECK: phi i64
; CHECK: !llvm.loop ![[LOOP3:[0-9]+]]
; CHECK: call i64 @__kittimer_stop(ptr %[[EPOCH3]])
; CHECK: sync within
; CHECK: ret void
;
; CHECK-DAG: ![[LOOP1]] = distinct !{![[LOOP1]], !{{[^,]+}}, ![[NAME1:[0-9]+]]}
; CHECK-DAG: ![[LOOP2]] = distinct !{![[LOOP2]], !{{[^,]+}}, ![[NAME2:[0-9]+]]}
; CHECK-DAG: ![[LOOP3]] = distinct !{![[LOOP3]], !{{[^,]+}}, ![[NAME3:[0-9]+]]}
; CHECK-DAG: ![[NAME1]] = !{!"tapir.loop.name", !"loop1"}
; CHECK-DAG: ![[NAME2]] = !{!"tapir.loop.name", !"loop2"}
; CHECK-DAG: ![[NAME3]] = !{!"tapir.loop.name", !"loop3"}

define void @generic(i64 %n) {
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
  br i1 %cmp1, label %exit1, label %header1, !llvm.loop !0

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
  br i1 %cmp2, label %exit2, label %header2, !llvm.loop !3

exit2:
  sync within %sr, label %ph3

ph3:
  br label %header3

header3:
  %i3 = phi i64 [ 0, %ph3 ], [ %next3, %latch3 ]
  detach within %sr, label %body3, label %latch3

body3:
  reattach within %sr, label %latch3

latch3:
  %next3 = add i64 %i3, 1
  %cmp3 = icmp eq i64 %next3, %n
  br i1 %cmp3, label %exit3, label %header3, !llvm.loop !3

exit3:
  sync within %sr, label %ret

ret:
  ret void
}

; PAPI: @papi
; PAPI: br label %[[INSTR_START:.+]]
; PAPI-EMPTY:
; PAPI-NEXT: [[INSTR_START]]:
; PAPI-NEXT: %[[EPOCH:.+]] = call ptr (ptr, i64, ...) @__kitpapi_start(ptr @[[PAPI]], i64 0)
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

; TIMER: @timer
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
!1 = !{!"tapir.loop.target", i32 512}
!2 = !{!"tapir.loop.name", !"loop1"}
!3 = distinct !{!3, !4, !5}
!4 = !{!"tapir.loop.target", i32 512}
!5 = !{!"tapir.loop.name", !"loop2"}
!6 = distinct !{!6, !7, !8}
!7 = !{!"tapir.loop.target", i32 512}
!8 = !{!"tapir.loop.name", !"loop3"}
