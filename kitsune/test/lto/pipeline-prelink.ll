; None of the Kitsune, or Tapir, passes should run during the prelink phase,
; regardless of the specified optimization level.
;
; -----------------------------------------------------------------------------
; Only the nolo tapir target is allowed at -O0.
;
; RUN: %kitcc -O2 --tapir=nolo -c -emit-llvm -o /dev/null %s \
; RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
; RUN:     | FileCheck %s
;
; -----------------------------------------------------------------------------
;
; RUN: %kitcc -O2 --tapir=serial -c -emit-llvm -o /dev/null %s \
; RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
; RUN:     | FileCheck %s
;
; RUN: %kitcc -O3 --tapir=serial -c -emit-llvm -o /dev/null %s \
; RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
; RUN:     | FileCheck %s
;
; RUN: %kitcc -Os --tapir=serial -c -emit-llvm -o /dev/null %s \
; RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
; RUN:     | FileCheck %s
;
; RUN: not %kitcc -Oz --tapir=serial -c -emit-llvm -o /dev/null %s \
; RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
; RUN:     | FileCheck %s --check-prefix=ERROR
;
; -----------------------------------------------------------------------------
;
; CHECK:      Running pass:      EarlyVerificationPass
; CHECK:      Running pass:      EarlyAnnotatePass
; CHECK:      Running pass:      NormalizeLoopControlBlocksPass
; CHECK:      Running pass:      SecondaryIVEliminationPass
; CHECK:      Running pass:      PrepareTapirLoopsPass
; CHECK:      Running pass:      LowerKitReduceIntrinsicsPass
; CHECK-NOT:  Running pass:      InstrumentPass
;
; CHECK-NOT:  Running pass:      PreLowerPreparePass
; CHECK-NOT:  Running pass:      SecondaryIVEliminationPass
; CHECK-NOT:  Running pass:      DeLICMPass
; CHECK-NOT:  Running pass:      PreLowerVerificationPass
; CHECK-NOT:  Running pass:      PreLowerAnnotatePass
; CHECK-NOT:  Running pass:      SerializePass
; CHECK-NOT:  Running pass:      LoopSpawningPass
; CHECK-NOT:  Running pass:      HoistAllocasPass
; CHECK-NOT:  Running pass:      EmbHoistAllocasPass
; CHECK-NOT:  Running pass:      LowerKitWarpIntrinsicsPass
; CHECK-NOT:  Running pass:      EmbResolveLibDeviceCallsPass
; CHECK-NOT:  Running pass:      EmbPreparePass
; CHECK-NOT:  Running pass:      EmbLinkLibDeviceBitcodePass
; CHECK-NOT:  Running pass:      EmbOptimizePass
; CHECK-NOT:  Running pass:      RecomputeKernelPropertiesPass
; CHECK-NOT:  Running pass:      GenerateCtorsPass
; CHECK-NOT:  Running pass:      LowerRuntimeIntrinsicsPass
;
; ERROR: unsupported optimization level '-Oz'
;
; -----------------------------------------------------------------------------
; The instrumentation pass will only run if instrumentation is explicitly
; enabled.
;
; RUN: %kitcc -O2 --tapir=serial -c -emit-llvm -o /dev/null %s \
; RUN:     --kit-instr=generic \
; RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
; RUN:     | FileCheck %s --check-prefix=INSTR
;
; RUN: %kitcc -O3 --tapir=serial -c -emit-llvm -o /dev/null %s \
; RUN:     --kit-instr=timer \
; RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
; RUN:     | FileCheck %s --check-prefix=INSTR
;
; RUN: %kitcc -Os --tapir=serial -c -emit-llvm -o /dev/null %s \
; RUN:     --kit-instr=timer,generic \
; RUN:     -flto -Xclang -fdebug-pass-manager 2>&1 \
; RUN:     | FileCheck %s --check-prefix=INSTR
;
; INSTR:      Running pass:      NormalizeLoopControlBlocksPass
; INSTR:      Running pass:      SecondaryIVEliminationPass
; INSTR:      Running pass:      PrepareTapirLoopsPass
; INSTR:      Running pass:      LowerKitReduceIntrinsicsPass
; INSTR:      Running pass:      InstrumentPass
;
; -----------------------------------------------------------------------------

declare void @ext(i64)

define void @f(i64 %n) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc.i, %latch ]
  detach within %syncreg, label %body, label %latch

body:
  call void @ext(i64 %i)
  reattach within %syncreg, label %latch

latch:
  %inc.i = add i64 %i, 1
  %cmp = icmp eq i64 %inc.i, %n
  br i1 %cmp, label %exit, label %header, !llvm.loop !0

exit:
  sync within %syncreg, label %end

end:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"tapir.loop.spawn.strategy", i32 1}
!3 = !{!"tapir.loop.lowering.enabled"}
