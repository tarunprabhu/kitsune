; None of the Kitsune, or Tapir, passes should run during the prelink phase,
; regardless of the specified optimization level.
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
; CHECK:      Running pass:      SerializePass
; CHECK:      Running pass:      PrepareTapirLoopsPass
; CHECK:      Running pass:      LocalizeReductionsPass
; CHECK:      Running pass:      ComputeReductionStrategyPass
; CHECK:      Running pass:      SecondaryIVEliminationPass
; CHECK-NOT:  Running pass:      InstrumentPass
;
; CHECK-NOT:  Running pass:      SecondaryIVEliminationPass
; CHECK-NOT:  Running pass:      PreLowerVerificationPass
; CHECK-NOT:  Running pass:      PreLowerAnnotatePass
; CHECK-NOT:  Running pass:      TapirLoopNormalizePass
; CHECK-NOT:  Running pass:      LoopSpawningPass
; CHECK-NOT:  Running pass:      HoistAllocasPass
; CHECK-NOT:  Running pass:      EmbHoistAllocasPass
; CHECK-NOT:  Running pass:      EmbLowerIntrinsicsPass
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
; INSTR:      Running pass:      PrepareTapirLoopsPass
; INSTR:      Running pass:      InstrumentPass
;
; -----------------------------------------------------------------------------

declare void @ext(i64)

define void @main(i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  %cmpn = icmp ult i64 0, %n
  br i1 %cmpn, label %ph, label %sync, !llvm.loop !0

ph:
  br label %header

header:
  %i = phi i64 [ 0, %ph ], [ %inc, %latch ]
  detach within %sr, label %reattach, label %latch

reattach:
  call void @ext(i64 %i)
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  br label %header

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"tapir.loop.spawn.strategy", i32 1}
!3 = !{!"tapir.loop.perfect.depth", i32 1}
!4 = !{!"tapir.loop.perfect.level", i32 1}
!5 = !{!"tapir.loop.lowering.enabled"}
