; Global variables with the kit_fb attribute must have external linkage.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: singleton fat binary global must have external linkage

@__hip_fatbin = internal constant [0 x i8] zeroinitializer, section ".hip_fatbin" #0

attributes #0 = { kit_fb kit_tt(4) }
