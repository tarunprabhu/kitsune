; Global variables with the kit_fb attribute must have an initializer.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: missing initializer in global containing fat binary

@fb = external global [0 x i8] #0

attributes #0 = { kit_fb kit_tt(4) }
