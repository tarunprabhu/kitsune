; Global variables with the kit_fb attribute must have a specific name.
;
; RUN: not llvm-as -o /dev/null %s 2>&1 | FileCheck %s
;
; CHECK: singleton fat binary global must be named __hip_fatbin

@.hip_fatbin = external constant [0 x i8], section ".hip_fatbin" #1

attributes #1 = { kit_fb kit_tt(4) }
