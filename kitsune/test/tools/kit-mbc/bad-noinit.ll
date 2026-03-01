; If a global variable that is expected to contain embedded bitcode was found,
; it must have an initializer.
;
; RUN: not %kit-mbc -S -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: initializer missing in global containing embedded bitcode

@bc = external global [2 x i8] #0

attributes #0 = { kit_bc kit_tt(2) }
