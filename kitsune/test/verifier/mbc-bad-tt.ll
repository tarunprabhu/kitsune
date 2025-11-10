; The value of the kit_tt attribute on an embedded bitcode global variable must
; be a tapir target that generates embedded bitcode. It is unlikely that the
; serial tapir target will ever generate embedded bitcode.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | sed 's/kit_bc kit_tt(2)/kit_bc kit_tt(1)/g' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: invalid value for 'kit_tt' attribute.
; CHECK-SAME: Tapir target does not generate embedded bitcode
