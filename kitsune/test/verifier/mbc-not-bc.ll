; Check that the correct error is emitted if the embedded bitcode global
; variable does not contain bitcode.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: invalid data in global containing embedded bitcode

@fb = constant [0 x i8] zeroinitializer #0
@bc = constant [4 x i8] c"BC\C0\DF" #1

attributes #0 = { kit_fb kit_tt(4) }
attributes #1 = { kit_bc kit_tt(4) }
