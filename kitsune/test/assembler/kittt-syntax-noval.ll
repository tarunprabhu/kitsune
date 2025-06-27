; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --check-prefix NOVAL
;
; NOVAL: expected integer

attributes #0 = { kit_tt() }
