; The type of global variables with the kit_fb attribute must be [n x i8] where
; N could be any positive integer
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: incorrect type of global containing fat binary

@fb = constant i256 zeroinitializer #0

attributes #0 = { kit_fb kit_tt(4) }
