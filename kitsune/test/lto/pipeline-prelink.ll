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
; CHECK:      Running pass:      PrepareTapirLoopsPass
; CHECK:      Running pass:      LowerKitReduceIntrinsicsPass
;
; CHECK-NOT:  Running pass:      PreLowerPreparePass
; CHECK-NOT:  Running pass:      SecondaryIVEliminationPass
; CHECK-NOT:  Running pass:      DeLICMPass
; CHECK-NOT:  Running pass:      PreLowerVerificationPass
; CHECK-NOT:  Running pass:      PreLowerAnnotatePass
; CHECK-NOT:  Running pass:      SerializePass
; CHECK-NOT:  Running pass:      LoopSpawningPass
; CHECK-NOT:  Running pass:      EmbResolveLibDeviceCallsPass
; CHECK-NOT:  Running pass:      EmbPreparePass
; CHECK-NOT:  Running pass:      EmbLinkLibDeviceBitcodePass
; CHECK-NOT:  Running pass:      EmbOptimizePass
; CHECK-NOT:  Running pass:      RecomputeKernelPropertiesPass
; CHECK-NOT:  Running pass:      GenerateCtorsPass
; CHECK-NOT:  Running pass:      LowerRuntimeIntrinsicsPass
;
; ERROR: unsupported optimization level '-Oz'

define void @f() {
  ret void
}
