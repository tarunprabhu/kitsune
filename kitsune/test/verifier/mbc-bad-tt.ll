; The value of the `bit.code` attribute must be a tapir target that generates
; embedded bitcode. It is unlikely that the serial tapir target will ever
; generate embedded bitcode.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | sed 's/i32 2/i32 1/g' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: invalid value for 'bit.code' attribute.
; CHECK-SAME: Tapir target does not generate embedded bitcode
