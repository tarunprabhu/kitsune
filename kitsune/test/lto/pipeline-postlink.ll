; -----------------------------------------------------------------------------
; If the tapir target is nolo, the behavior is similar to the regular pipeline
; i.e. loop spawning is not run and neither are any Kitsune passes.
;
; RUN: %kitcc -flto -O2 --tapir=nolo -o /dev/null %s %sysroot \
; RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
; RUN:     | FileCheck %s -check-prefix NOLO
;
; NOLO:      Running pass:     VerifierPass
; NOLO-NOT:  Running pass:     LoopSpawning
; NOLO:      Running pass:     VerifierPass
; NOLO-NEXT: Running analysis: VerifierAnalysis
;
; -----------------------------------------------------------------------------
; The Kitsune (and Tapir) lowering passes should run during the postlink phase
; of LTO. But the non-lowering passes should not run.
;
; RUN: %kitcc -flto -O2 --tapir=serial -o /dev/null %s %sysroot \
; RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
; RUN:     | FileCheck %s -check-prefix O23S
;
; RUN: %kitcc -flto -O3 --tapir=serial -o /dev/null %s %sysroot \
; RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
; RUN:     | FileCheck %s -check-prefix O23S
;
; RUN: %kitcc -flto -Os --tapir=serial -o /dev/null %s %sysroot \
; RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
; RUN:     | FileCheck %s -check-prefix O23S
;
; RUN: not %kitcc -flto -Oz --tapir=serial -o /dev/null %s %sysroot \
; RUN:     -Xlinker --lto-debug-pass-manager -Xlinker --lto-emit-llvm 2>&1 \
; RUN:     | FileCheck %s -check-prefix ERROR
;
; -----------------------------------------------------------------------------
;
; O23S-NOT:   Running pass:      EarlyVerificationPass
; O23S-NOT:   Running pass:      EarlyAnnotatePass
; O23S-NOT:   Running pass:      SerializePass
; O23S-NOT:   Running pass:      PrepareTapirLoopsPass
; O23S-NOT:   Running pass:      LocalizeReductionsPass
; O23S-NOT:   Running pass:      ComputeReductionStrategyPass
;
; O23S:       Running pass:      SecondaryIVEliminationPass
; O23S:       Running pass:      SimplifyCFGPass
; O23S:       Running pass:      LoopSimplifyPass
; O23S:       Running pass:      PreLowerVerificationPass
; O23S:       Running pass:      PreLowerAnnotatePass
; O23S:       Running pass:      TapirLoopNormalizePass
; O23S:       Running pass:      LoopSpawningPass
; O23S:       Running pass:      HoistAllocasPass
; O23S:       Running pass:      EmbHoistAllocasPass
; O23S:       Running pass:      TapirToTargetPass
; O23S:       Running pass:      PrefetchForDevicePass
; O23S:       Running pass:      EmbFinalizeReductionsPass
; O23S:       Running pass:      EmbLowerIntrinsicsPass
; O23S:       Running pass:      EmbResolveLibDeviceCallsPass
; O23S:       Running pass:      EmbPreparePass
; O23S:       Running pass:      EmbLinkLibDeviceBitcodePass
; O23S:       Running pass:      EmbOptimizePass
; O23S:       Running pass:      RecomputeKernelPropertiesPass
; O23S:       Running pass:      GenerateCtorsPass
; O23S:       Running pass:      VerifierPass
; O23S:       Running analysis:  VerifierAnalysis
;
; ERROR: unsupported optimization level '-Oz'

declare void @ext(i64)

; This function must be named @main, otherwise it will be removed since there
; are no uses of this function in the module.
define void @main(i64 %n) {
entry:
  %sr = call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  detach within %sr, label %body, label %latch

body:
  call void @ext(i64 %i)
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %sync, label %header, !llvm.loop !2

sync:
  sync within %sr, label %exit

exit:
  ret void
}

!0 = !{i32 1, !"ThinLTO", i32 0}
!1 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!2 = distinct !{!2, !3, !4, !5, !6, !7, !8}
!3 = !{!"tapir.loop.target", i32 1}
!4 = !{!"tapir.loop.spawn.strategy", i32 1}
!5 = !{!"tapir.loop.perfect.depth", i32 1}
!6 = !{!"tapir.loop.perfect.level", i32 1}
!7 = !{!"tapir.loop.lowering.enabled"}
!8 = !{!"tapir.loop.prepared"}
