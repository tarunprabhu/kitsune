; Multiple fat binaries are allowed in a host module as long as they are from
; distinct tapir targets.
;
; RUN: llvm-as %s -o /dev/null 2>&1 | FileCheck %s --allow-empty
;
; CHECK-NOT: {{.+}}

@__nv_fatbin = external constant [0 x i8], section ".nv_fatbin" #0
@__hip_fatbin = external constant [0 x i8], section ".hip_fatbin" #1

attributes #0 = { kit_fb kit_tt(2) }
attributes #1 = { kit_fb kit_tt(4) }
