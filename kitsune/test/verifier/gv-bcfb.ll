; A global variable cannot have both the kit_bc and kit_fb attributes.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: Attributes 'kit_bc' and 'kit_fb' are incompatible

@g = constant [8 x i8] zeroinitializer #0

attributes #0 = { kit_bc kit_fb kit_tt(4) }
