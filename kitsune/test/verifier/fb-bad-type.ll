; The type of global variables with the kit_fb attribute must be [0 x i8]
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: singleton fat binary global must be a zero-length array of bytes

@__hip_fatbin = external constant [1 x i8], section ".hip_fatbin" #0

attributes #0 = { kit_fb kit_tt(4) }
