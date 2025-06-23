; The initializer of global variables with the kit_fb attribute must be either
; a constant array or a zero.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: invalid initializer in global containing fat binary

@fb = constant [1 x i8] undef #0

attributes #0 = { kit_fb(1) }
