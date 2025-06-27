; The initializer of global variables with the kit_kernel_props attribute must
; be either a constant struct or a zero.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: invalid initializer in global containing kernel properties

@0 = constant { i64, i64, i64, i64 } undef #0

attributes #0 = { kit_tt(2) "kit_kernel_props"="funcname" }
