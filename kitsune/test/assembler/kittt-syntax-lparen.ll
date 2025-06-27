; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --check-prefix LPAREN
;
; LPAREN: expected '('

attributes #1 = { kit_tt }
