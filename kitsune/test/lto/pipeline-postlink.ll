; -----------------------------------------------------------------------------
; If the tapir target is nolo, the behavior is similar to the regular pipeline
; i.e. loop spawning is not run and neither are any Kitsune passes.
;
; RUN: %kitcc -flto -O2 --tapir=nolo -o /dev/null %s %sysroot \
; RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
; RUN:     | FileCheck %s -check-prefix NOLO
;
; NOLO:        Running pass:       VerifierPass
; NOLO-NOT:    Running pass:       LoopSpawning
; NOLO:        Running pass:       VerifierPass
; NOLO-NEXT:   Running analysis:   VerifierAnalysis
;
; -----------------------------------------------------------------------------
; The Kitsune (and Tapir) lowering passes should run during the postlink phase
; of LTO. But the non-lowering passes should not run.
;
; RUN: %kitcc -flto -O1 --tapir=serial -o /dev/null %s %sysroot \
; RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123
;
; RUN: %kitcc -flto -O2 --tapir=serial -o /dev/null %s %sysroot \
; RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123
;
; RUN: %kitcc -flto -O3 --tapir=serial -o /dev/null %s %sysroot \
; RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123
;
; RUN: not %kitcc -flto -Os --tapir=serial -o /dev/null %s %sysroot \
; RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
; RUN:     | FileCheck %s -check-prefix ERROR
;
; RUN: not %kitcc -flto -Oz --tapir=serial -o /dev/null %s %sysroot \
; RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
; RUN:     | FileCheck %s -check-prefix ERROR
;
; -----------------------------------------------------------------------------
;
; O123-NOT:    Running pass:       EarlyVerificationPass
; O123-NOT:    Running pass:       EarlyAnnotatePass
; O123-NOT:    Running pass:       SerializePass
; O123-NOT:    Running pass:       GuardLoopPass
; O123-NOT:    Running pass:       PrepareTapirLoopsPass
; O123-NOT:    Running pass:       LocalizeReductionsPass
; O123-NOT:    Running pass:       InstrumentPass
;
; ... pipeline pre-tapir
;
; ... pipeline pre-loop-spawning
; O123:        Running pass:       SecondaryIVEliminationPass
; O123:        Running pass:       PreLowerVerificationPass
; O123:        Running pass:       PreLowerAnnotatePass
; O123:        Running pass:       TapirLoopNormalizePass
;
; O123:        Running pass:       LoopSpawningPass
;
; ... pipeline post-loop-spawning
; O123:        Running pass:       HoistAllocasPass
; O123:        Running pass:       EmbHoistAllocasPass
;
; O123:        Running pass:       TapirToTargetPass
;
; ... pipeline post-tapir
; O123:        Running pass:       PrefetchForDevicePass
; O123:        Running pass:       EmbLowerIntrinsicsPass
; O123:        Running pass:       EmbResolveLibDeviceCallsPass
; O123:        Running pass:       EmbPreparePass
; O123:        Running pass:       EmbLinkLibDeviceBitcodePass
; O123:        Running pass:       EmbOptimizePass
; O123:        Running pass:       RecomputeKernelPropertiesPass
; O123:        Running pass:       GenerateCtorsPass
;
; O123:        Running pass:       VerifierPass
; O123:        Running analysis:   VerifierAnalysis
;
; ERROR: unsupported optimization level '-O{{[sz]}}'

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %n = sext i32 %argc to i64
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
  ret i32 0
}

declare void @ext(i64)

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"tapir.loop.spawn.strategy", i32 1}
!3 = !{!"tapir.loop.lowering.enabled"}
