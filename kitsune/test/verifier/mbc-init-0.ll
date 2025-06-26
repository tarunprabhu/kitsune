; The initializer of global variables with the kit_bc attribute cannot be zero
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: invalid initializer in global containing bitcode

@bc = constant [1 x i8] zeroinitializer #0
@fb = constant [0 x i8] zeroinitializer #1

attributes #0 = { kit_bc kit_tt(1) }
attributes #1 = { kit_fb kit_tt(1) }
