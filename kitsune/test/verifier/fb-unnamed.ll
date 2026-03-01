; The global containing the device code must be named. Check that the verifier
; fails if it is not.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: global containing device code does not have a name

@0 = constant [0 x i8] zeroinitializer #0

attributes #0 = { kit_fb kit_tt(2) }
