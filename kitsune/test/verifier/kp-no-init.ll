; Global variables with the kit_kernel_props attribute must have an initializer.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: missing initializer in global containing kernel properties

@0 = external global { i64, i64, i64, i64 } #0

attributes #0 = { "kit_kernel_props"="f" kit_tt(4) }
