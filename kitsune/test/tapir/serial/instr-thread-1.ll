; Check that generic instrumentation is correctly added inside a tapir loop.
;
; RUN: opt -passes="kit-instrument" -S %s \
; RUN:     --kit-instr-unit=thread  --kit-instr=generic 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: cannot instrument threads in loop with tapir target 'serial'
; CHECK-NEXT: from loop 'serial'
;
; CHECK-LABEL: @f
; CHECK-NOT: __kit_instr_start
; CHECK-NOT: __kit_instr_stop

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
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"tapir.loop.name", !"serial"}
