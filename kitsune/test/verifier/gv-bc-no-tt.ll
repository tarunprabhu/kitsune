; A global variable with the kit_bc attribute must have the kit_tt attribute.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | sed 's/kit_bc kit_tt(4)/kit_bc/g' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: Attribute 'kit_bc' requires 'kit_tt'
