; A global variable with the kit_fb attribute must have the kit_tt attribute.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: Attribute 'kit_fb' requires 'kit_tt'

@.kitsune.emb.fb = constant [0 x i8] zeroinitializer #0

attributes #0 = { kit_fb }
