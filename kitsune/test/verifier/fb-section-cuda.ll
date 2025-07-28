; Global variables with the kit_fb attribute must be in a specific section.
;
; RUN: not llvm-as -o /dev/null %s 2>&1 | FileCheck %s
;
; CHECK: singleton fat binary global must be in section .nv_fatbin

@__nv_fatbin = external constant [0 x i8], section "__nv_fatbin" #1

attributes #1 = { kit_fb kit_tt(2) }
