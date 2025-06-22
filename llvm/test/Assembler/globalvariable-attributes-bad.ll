; RUN: not llvm-as < %s 2>&1 | FileCheck %s
;
; CHECK: this attribute does not apply to global variables

@g = global i32 2 #0

attributes #0 = { norecurse }

