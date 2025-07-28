; Global variables with the kit_fb attribute must not have an initializer
;
; RUN: llvm-as %s -o /dev/null 2>&1 2>&1 | FileCheck %s --allow-empty
;
; CHECK-NOT: {{.+}}

@__nv_fatbin = external constant [0 x i8], section ".nv_fatbin" #0

attributes #0 = { kit_fb kit_tt(2) }
