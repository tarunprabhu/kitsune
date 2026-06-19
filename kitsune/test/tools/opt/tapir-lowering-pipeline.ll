; Check that the tapir-lowering meta-pass adds the expected passes to the
; pipeline.
;
; ------------------------------------------------------------------------------
; Tapir lowering is available at O0, but only a limited set of passes are run.
;
; RUN: opt -passes='tapir-lowering<O0>' --tapir=serial -debug-pass-manager %s \
; RUN:     -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O0
;
; O0:         Running pass:     TapirToTargetPass
; O0:         Running pass:     VerifierPass
;
; ------------------------------------------------------------------------------
; At higher optimization levels, the Kitsune passes that are run are always
; the same. If we ever have optimization passes that are dependent on the
; optimization level, this should be updated
;
; RUN: opt -passes='tapir-lowering<O1>' --tapir=serial -debug-pass-manager %s \
; RUN:     -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123S
;
; RUN: opt -passes='tapir-lowering<O2>' --tapir=serial -debug-pass-manager %s \
; RUN:     -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123S
;
; RUN: opt -passes='tapir-lowering<O3>' --tapir=serial -debug-pass-manager %s \
; RUN:     -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123S
;
; RUN: opt -passes='tapir-lowering<Os>' --tapir=serial -debug-pass-manager %s \
; RUN:     -disable-output 2>&1 \
; RUN:     | FileCheck %s -check-prefix O123S
;
; RUN: not opt -passes='tapir-lowering<Oz>' --tapir=serial -debug-pass-manager \
; RUN:     -disable-output %s 2>&1 \
; RUN:     | FileCheck %s -check-prefix ERROR
;
; ERROR: unsupported optimization level '-Oz'
;
; O123S:      Running pass:     PreLowerVerificationPass
; O123S:      Running pass:     PreLowerAnnotate
; O123S:      Running pass:     SerializePass
; O123S:      Running pass:     LoopSpawningPass
; O123S:      Running pass:     TapirToTargetPass
; O123S:      Running pass:     IPSCCPPass
; O123S:      Running pass:     CalledValuePropagationPass
; O123S:      Running pass:     GlobalOptPass
; O123S:      Running pass:     DeadArgumentEliminationPass
; O123S:      Running pass:     AlwaysInlinerPass
; O123S:      Running pass:     EliminateAvailableExternallyPass
; O123S:      Running pass:     ReversePostOrderFunctionAttrs
; O123S:      Running pass:     GlobalDCEPass
; O123S:      Running pass:     VerifierPass
;
; ------------------------------------------------------------------------------

define void @f() {
  ret void
}
