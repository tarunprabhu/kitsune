; The embedded bitcode in this file contains an embedded bitcode global. This
; is not allowed.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | %kit-enc --tapir=cuda \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: embedded module cannot contain embedded bitcode
;
; NOTE: The embedded bitcode in this embedded bitcode is actually valid. Right
; now, it doesn't make a difference because the verifier will fail if it finds
; any global variable with the kit_bc attribute in the embedded module. Just to
; avoid having this test spuriously fail if the verifier gets stricter, we must
; ensure that *all* the bitcode is valid - all the way down.
