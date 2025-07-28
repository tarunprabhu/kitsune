; Global variables with the kit_fb attribute must have a specific name.
;
; RUN: not llvm-as -o /dev/null %s 2>&1 | FileCheck %s
;
; CHECK: singleton fat binary global must be named __nv_fatbin

@.nv_fatbin = external constant [0 x i8], section ".nv_fatbin" #0

attributes #0 = { kit_fb kit_tt(2) }
