; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --check-prefix TOO-LARGE
;
; TOO-LARGE: expected 32-bit integer (too large)

attributes #0 = { kit_tt(98034239282911) }
