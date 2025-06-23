; A global variable cannot have both the kit_fb and kit_kernel_props attributes.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: Attributes 'kit_fb' and 'kit_kernel_props' are incompatible

@.kitsune.emb.fb = constant [0 x i8] zeroinitializer #0

attributes #0 = { kit_fb(1) "kit_kernel_props"="some_kernel" }
