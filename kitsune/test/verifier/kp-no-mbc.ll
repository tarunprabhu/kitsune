; It is ok to have a global variable with the kit_kernel_props attribute that
; does not have a corresponding embedded bitcode global. The global containing
; embedded bitcode will be removed after the fat binary is generated, but the
; global with the kernel properties will not.
;
; RUN: llvm-as %s -o /dev/null

@0 = constant { i64, i64, i64, i64 } zeroinitializer #0
attributes #0 = { kit_tt(2) "kit_kernel_props"="f" }
