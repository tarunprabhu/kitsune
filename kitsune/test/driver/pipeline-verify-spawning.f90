! REQUIRES: kitfc
!
! If -kit-verify-spawning is given, the verifier pass must be run immediately
! after the task and loop spawning passes.
!
! RUN: %kitfc --tapir=serial -c -O1 -o /dev/null %s \
! RUN:     -Xclang -fdebug-pass-manager 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=ALL,NOVERIFY
!
! RUN: %kitfc --tapir=serial -c -O1 -o /dev/null %s \
! RUN:     -Xclang -fdebug-pass-manager -mllvm -kit-verify-spawning 2>&1 \
! RUN:     | FileCheck %s --check-prefixes=ALL,VERIFY
!
! ALL:             Running pass:       TapirLoopNormalizePass
! ALL-NEXT:        Running pass:       TaskSpawningPass
! ALL-NEXT:        Running pass:       LoopSpawningPass
! NOVERIFY-NOT:    Running pass:       VerifierPass
! VERIFY-NEXT:     Running pass:       VerifierPass
! ALL:             Running pass:       HoistAllocasPass

subroutine f()
end subroutine f
