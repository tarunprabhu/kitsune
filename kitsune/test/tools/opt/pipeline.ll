; -----------------------------------------------------------------------------
; If the --tapir option is not provided to opt, neither tapir, nor Kitsune
; passes are run.
;
; RUN: opt -O0 -debug-pass-manager %s -disable-output 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: opt -O1 -debug-pass-manager %s -disable-output 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: opt -O2 -debug-pass-manager %s -disable-output 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: opt -O3 -debug-pass-manager %s -disable-output 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: opt -Os -debug-pass-manager %s -disable-output 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: opt -Oz -debug-pass-manager %s -disable-output 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; DEFAULT-NOT: Running pass:     SerializePass
; DEFAULT-NOT: Running pass:     PrepareTapirLoopsPass
; DEFAULT-NOT: Running pass:     ComputeReductionStrategyPass
; DEFAULT-NOT: Running pass:     PreLowerAnnotatePass
; DEFAULT-NOT: Running pass:     LoopSpawningPass
; DEFAULT-NOT: Running pass:     HoistAllocasPass
; DEFAULT-NOT: Running pass:     EmbHoistAllocasPass
; DEFAULT-NOT: Running pass:     TapirToTargetPass
; DEFAULT-NOT: Running pass:     PrefetchForDevicePass
; DEFAULT-NOT: Running pass:     EmbFinalizeReductionsPass
; DEFAULT-NOT: Running pass:     EmbLowerIntrinsicsPass
; DEFAULT-NOT: Running pass:     EmbResolveLibDeviceCallsPass
; DEFAULT-NOT: Running pass:     EmbPreparePass
; DEFAULT-NOT: Running pass:     EmbLinkLibDeviceBitcodePass
; DEFAULT-NOT: Running pass:     EmbOptimizePass
; DEFAULT-NOT: Running pass:     RecomputeKernelPropertiesPass
; DEFAULT-NOT: Running pass:     GenerateCtorsPass
;
; -----------------------------------------------------------------------------
; Unlike the frontends, -O0 is allowed with --tapir, even if the tapir target
; is not nolo. In this case, only a limited number of passes are run.
; We don't use the text of this file as input here because it will result in a
; failure since loop-spawning will not run.
;
; FIXME: Should we consider not allowing -O0 as an optimization level if
; --tapir is given. It is not clear what advantage there is to allowing -O0 if
; it is likely to result in a failure.
;
; RUN: echo "" | opt -O0 --tapir=serial -debug-pass-manager -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix O0 %s
;
; O0:      Running pass:     TapirToTargetPass
; O0:      Running pass:     AlwaysInlinerPass
; O0:      Running pass:     VerifierPass
; O0:      Running pass:     BitcodeWriterPass
;
; -----------------------------------------------------------------------------
; If the --tapir option is provided to opt, the Kitsune passes are run at all
; optimization levels.
;
; RUN: opt -O1 --tapir=serial -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix O123S %s
;
; RUN: opt -O2 --tapir=serial -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix O123S %s
;
; RUN: opt -O3 --tapir=serial -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix O123S %s
;
; RUN: opt -Os --tapir=serial -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix O123S %s
;
; The Early* passes run early in the pass pipeline.
; O123S:       Running pass:     EarlyVerificationPass
; O123S:       Running pass:     EarlyAnnotatePass
;
; <KIT-PREPARE>
; O123S:       Running pass:     SerializePass
; O123S:       Running pass:     PrepareTapirLoopsPass
; O123S:       Running pass:     LocalizeReductionsPass
; O123S:       Running pass:     ComputeReductionStrategyPass
; O123S-NOT:   Running pass:     InstrumentPass
; </KIT-PREPARE>
;
; <KIT-PRE-VECTORIZE>
; O123S:       Running pass:     SecondaryIVEliminationPass
; </KIT-PRE-VECTORIZE>
;
; <KIT-PRE-TAPIR>
; There are no standard pre-tapir passes at this time
; </KIT-PRE-TAPIR>
;
; <KIT-PRE-LOOP-SPAWNING>
; O123S:       Running pass:     SecondaryIVEliminationPass
; O123S:       Running pass:     PreLowerVerificationPass
; O123S:       Running pass:     PreLowerAnnotatePass
; </KIT-PRE-LOOP-SPAWNING>
;
; O123S:       Running pass:     LoopSpawningPass
;
; <KIT-POST-LOOP-SPAWNING>
; O123S:       Running pass:     HoistAllocasPass
; O123S:       Running pass:     EmbHoistAllocasPass
; </KIT-POST-LOOP-SPAWNING>
;
; O123S:       Running pass:     TapirToTargetPass
; O123S:       Running pass:     GlobalDCEPass
;
; <KIT-POST-TAPIR>
; O123S:       Running pass:     PrefetchForDevicePass
; O123S:       Running pass:     EmbFinalizeReductionsPass
; O123S:       Running pass:     EmbLowerIntrinsicsPass
; O123S:       Running pass:     EmbResolveLibDeviceCallsPass
; O123S:       Running pass:     EmbPreparePass
; O123S:       Running pass:     EmbLinkLibDeviceBitcodePass
; O123S:       Running pass:     EmbOptimizePass
; O123S:       Running pass:     RecomputeKernelPropertiesPass
; O123S:       Running pass:     GenerateCtorsPass
; </KIT-POST-TAPIR>
;
; O123S:       Running pass:     VerifierPass
; O123S:       Running pass:     BitcodeWriterPass
;
; -----------------------------------------------------------------------------
; The instrumentation pass will only run if instrumentation is explicitly
; enabled.
;
; RUN: opt -O1 --tapir=serial -debug-pass-manager -o /dev/null %s \
; RUN:     --kit-instr=generic 2>&1 \
; RUN:     | FileCheck %s --check-prefix=INSTR
;
; RUN: opt -O2 --tapir=serial -debug-pass-manager -o /dev/null %s \
; RUN:     --kit-instr=papi 2>&1 \
; RUN:     | FileCheck %s --check-prefix=INSTR
;
; RUN: opt -O3 --tapir=serial -debug-pass-manager -o /dev/null %s \
; RUN:     --kit-instr=timer 2>&1 \
; RUN:     | FileCheck %s --check-prefix=INSTR
;
; RUN: opt -Os --tapir=serial -debug-pass-manager -o /dev/null %s \
; RUN:     --kit-instr=generic,timer 2>&1 \
; RUN:     | FileCheck %s --check-prefix=INSTR
;
; INSTR:      Running pass:      PrepareTapirLoopsPass
; INSTR:      Running pass:      InstrumentPass
;
; -----------------------------------------------------------------------------

define void @f(i64 %n) {
entry:
  %sr = tail call token @llvm.syncregion.start()
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %latch ]
  %cmp = icmp ne i64 %i, %n
  br i1 %cmp, label %detach, label %sync

detach:
  detach within %sr, label %reattach, label %latch

reattach:
  call void @ext(i64 %i)
  reattach within %sr, label %latch

latch:
  %inc = add i64 %i, 1
  br label %header, !llvm.loop !0

sync:
  sync within %sr, label %exit

exit:
  ret void
}

declare void @ext(i64)

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"tapir.loop.spawn.strategy", i32 1}
!3 = !{!"tapir.loop.perfect.depth", i32 1}
!4 = !{!"tapir.loop.perfect.level", i32 1}
!5 = !{!"tapir.loop.lowering.enabled"}
