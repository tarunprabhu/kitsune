; The global containing embedded bitcode must be named. Check that the verifier
; fails if it is not.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | sed 's/[.]kitsune[.]emb[.]bc/0/g' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: global containing embedded bitcode does not have a name

