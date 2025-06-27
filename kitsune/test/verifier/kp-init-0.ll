; The initializer of global variables with the kit_kernel_props attribute can be
; zero.
;
; RUN: llvm-as %s -o /dev/null

@0 = constant [1 x i8] zeroinitializer #0

attributes #0 = { kit_tt(4) "kit_kp_props"="kernel_func" }
