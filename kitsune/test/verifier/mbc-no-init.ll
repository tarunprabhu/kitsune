; Global variables with the kit_bc attribute must have an initializer.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: missing initializer in global containing bitcode

@bc = external global [0 x i8] #0
@fb = constant [0 x i8] zeroinitializer #1

attributes #0 = { kit_bc kit_tt(4) }
attributes #1 = { kit_fb kit_tt(4) }
