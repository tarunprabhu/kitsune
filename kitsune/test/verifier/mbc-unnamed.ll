; The global containing embedded bitcode must be named. Check that the verifier
; fails if it is not.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | sed 's/[.]kit[.]emb[.]bc/0/g' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | sed 's/[.]kit[.]emb[.]bc/0/g' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: global with attribute 'kit.gv.bit.code': missing required name
; CHECK-NEXT: from global variable '@0'

