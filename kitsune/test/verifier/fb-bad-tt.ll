; The value of the kit_tt attribute on a fat binary global variable must be
; a tapir target that generates embedded bitcode. It is unlikely that the
; serial tapir target will ever generate embedded bitcode.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: invalid value for 'kit_tt' attribute. Tapir target does not generate embedded bitcode

@0 = constant [0 x i8] zeroinitializer #0

attributes #0 = { kit_fb kit_tt(1) }
