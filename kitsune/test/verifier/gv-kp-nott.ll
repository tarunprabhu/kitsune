; A global variable with the kit_kernel_props attribute must have the kit_tt
; attribute.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: Attribute 'kit_kernel_props' requires 'kit_tt'

@.kitsune.emb.fb = constant { i64, i64, i64, i64 } zeroinitializer #0

attributes #0 = { "kit_kernel_props"="some_kernel_name" }
