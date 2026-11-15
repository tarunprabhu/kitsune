; If -kit-verify-spawning is given, the verifier pass must be run immediately
; after the task and loop spawning passes.
;
; RUN: opt --tapir=serial -passes='kit-lowering<O1>' \
; RUN:     -debug-pass-manager -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=ALL,NOVERIFY
;
; RUN: opt --tapir=serial -passes='kit-lowering<O1>' -kit-verify-spawning \
; RUN:     -debug-pass-manager -disable-output %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=ALL,VERIFY
;
; ALL:             Running pass:       TapirLoopNormalizePass
; ALL-NEXT:        Running pass:       TaskSpawningPass
; ALL-NEXT:        Running pass:       LoopSpawningPass
; NOVERIFY-NOT:    Running pass:       VerifierPass
; VERIFY-NEXT:     Running pass:       VerifierPass
; ALL:             Running pass:       HoistAllocasPass

define void @f() {
  ret void
}
