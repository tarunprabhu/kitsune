; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --check-prefix NOTINT
;
; NOTINT: expected integer

attributes #1 = { kit_tt(cuda) }
