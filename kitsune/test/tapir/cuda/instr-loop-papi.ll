; Check that PAPI instrumentation is not added around a tapir loop to be run
; on an NVIDIA GPU.
;
; RUN: opt -passes="kit-instrument" -S %s \
; RUN:     --kit-instr-unit=loop  --kit-instr=papi 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: cannot add PAPI instrumentation to loop with tapir target 'cuda'
; CHECK-NEXT: from loop 'cuda'
;
; CHECK-LABEL: @f
; CHECK-NOT: __kit_papi_start
; CHECK-NOT: __kit_papi_stop

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
!1 = !{!"tapir.loop.target", i32 2}
!2 = !{!"tapir.loop.name", !"cuda"}
