; Global variables with the kit_fb attribute must have a name.
;
; RUN: not llvm-as -o /dev/null %s 2>&1 | FileCheck %s
;
; CHECK: singleton fat binary global must have a name

@0 = external constant [0 x i8], section ".hip_fatbin" #1

attributes #1 = { kit_fb kit_tt(4) }

