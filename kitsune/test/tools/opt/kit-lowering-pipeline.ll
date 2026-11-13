; Check that the Kitsune-lowering meta-pass adds the expected passes to the
; pipeline.
;
; ------------------------------------------------------------------------------
; Kitsune lowering is available at O0, but only a limited set of passes are run.
; We don't use the text of this file as input here because it will result in a
; failure since loop-spawning will not run.
;
; FIXME: Should we consider not allowing -O0 as an optimization level for
; Kitsune's lowering passes?
;
; RUN: echo "" \
; RUN:     | opt --tapir=serial -passes='kit-lowering<O0>' -debug-pass-manager \
; RUN:           -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O0
;
; O0:          Running pass:       TapirToTargetPass
; O0:          Running pass:       AlwaysInlinerPass
; O0:          Running pass:       VerifierPass
;
; ------------------------------------------------------------------------------
; At higher optimization levels, the Kitsune passes that are run are always
; the same. This may need to change if we ever have optimization-level-dependent
; Kitsune lowering pipelines.
;
; RUN: opt --tapir=serial -passes='kit-lowering<O1>' -debug-pass-manager %s \
; RUN:     -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123
;
; RUN: opt --tapir=serial -passes='kit-lowering<O2>' -debug-pass-manager %s \
; RUN:     -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123
;
; RUN: opt --tapir=serial -passes='kit-lowering<O3>' -debug-pass-manager %s \
; RUN:     -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123
;
; RUN: not opt --tapir=serial -passes='kit-lowering<Os>' -debug-pass-manager \
; RUN:     %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix ERROR
;
; RUN: not opt --tapir=serial -passes='kit-lowering<Oz>' -debug-pass-manager \
; RUN:     %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s -check-prefix ERROR
;
; ERROR: The optimization level "{{.+}}" is no longer supported
;
; O123-NOT:    Running pass:       EarlyVerificationPass
; O123-NOT:    Running pass:       EarlyAnnotatePass
; O123-NOT:    Running pass:       SerializePass
; O123-NOT:    Running pass:       GuardLoopPass
; O123-NOT:    Running pass:       PrepareTapirLoops
; O123-NOT:    Running pass:       LocalizeReductionsPass
; O123-NOT:    Running pass:       SecondaryIVEliminationPass
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
; O123Z:       Running pass:       HoistAllocasPass
; O123Z:       Running pass:       EmbHoistAllocasPass
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
; </KIT-POST-TAPIR>
;
; O123:        Running pass:       VerifierPass
; O123:        Running pass:       BitcodeWriterPass

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

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"tapir.loop.target", i32 1}
!2 = !{!"tapir.loop.spawn.strategy", i32 1}
!3 = !{!"tapir.loop.lowering.enabled"}
!4 = !{!"tapir.loop.perfect.level", i32 1}
!5 = !{!"tapir.loop.perfect.depth", i32 1}
