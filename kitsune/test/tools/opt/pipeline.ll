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
; RUN: not opt -Os -debug-pass-manager %s -disable-output 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; RUN: not opt -Oz -debug-pass-manager %s -disable-output 2>&1 \
; RUN:     | FileCheck -check-prefix DEFAULT %s
;
; DEFAULT-NOT: Running pass:       EarlyVerificationPass
; DEFAULT-NOT: Running pass:       EarlyAnnotatePass
; DEFAULT-NOT: Running pass:       SerializePass
; DEFAULT-NOT: Running pass:       GuardLoopPass
; DEFAULT-NOT: Running pass:       PrepareTapirLoopsPass
; DEFAULT-NOT: Running pass:       LocalizeReductionsPass
; DEFAULT-NOT: Running pass:       PreLowerAnnotatePass
; DEFAULT-NOT: Running pass:       LoopSpawningPass
; DEFAULT-NOT: Running pass:       HoistAllocasPass
; DEFAULT-NOT: Running pass:       EmbHoistAllocasPass
; DEFAULT-NOT: Running pass:       TapirToTargetPass
; DEFAULT-NOT: Running pass:       PrefetchForDevicePass
; DEFAULT-NOT: Running pass:       EmbLowerIntrinsicsPass
; DEFAULT-NOT: Running pass:       EmbResolveLibDeviceCallsPass
; DEFAULT-NOT: Running pass:       EmbPreparePass
; DEFAULT-NOT: Running pass:       EmbLinkLibDeviceBitcodePass
; DEFAULT-NOT: Running pass:       EmbOptimizePass
; DEFAULT-NOT: Running pass:       RecomputeKernelPropertiesPass
; DEFAULT-NOT: Running pass:       GenerateCtorsPass
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
; O0:          Running pass:       TapirToTargetPass
; O0:          Running pass:       AlwaysInlinerPass
; O0:          Running pass:       VerifierPass
; O0:          Running pass:       BitcodeWriterPass
;
; -----------------------------------------------------------------------------
; If the --tapir option is provided to opt, the Kitsune passes are run at all
; optimization levels.
;
; RUN: opt -O1 --tapir=serial -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix O123 %s
;
; RUN: opt -O2 --tapir=serial -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix O123 %s
;
; RUN: opt -O3 --tapir=serial -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix O123 %s
;
; RUN: not opt -Os --tapir=serial -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix ERROR %s
;
; RUN: not opt -Oz --tapir=serial -debug-pass-manager %s -o /dev/null 2>&1 \
; RUN:     | FileCheck -check-prefix ERROR %s
;
; ... pipeline early
; O123:        Running pass:       EarlyVerificationPass
; O123:        Running pass:       EarlyAnnotatePass
;
; ... pipeline prepare
; O123:        Running pass:       SerializePass
; O123:        Running pass:       GuardLoopPass
; O123:        Running pass:       PrepareTapirLoopsPass
; O123:        Running pass:       LocalizeReductionsPass
; O123-NOT:    Running pass:       InstrumentPass
;
; ... pipeline pre-vectorize
; O123:        Running pass:       SecondaryIVEliminationPass
;
; ... pipeline pre-tapir
; There are no standard pre-tapir passes at this time
;
; ... pipeline pre-loop-spawning
; O123:        Running pass:       SecondaryIVEliminationPass
; O123:        Running pass:       PreLowerVerificationPass
; O123:        Running pass:       PreLowerAnnotatePass
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
; O123:        Running pass:       BitcodeWriterPass
;
; ERROR: The optimization level "{{.+}}" is no longer supported
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
; INSTR:       Running pass:       PrepareTapirLoopsPass
; INSTR:       Running pass:       InstrumentPass
;
; -----------------------------------------------------------------------------

define i32 @f(i32 %argc, ptr %argv) {
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
