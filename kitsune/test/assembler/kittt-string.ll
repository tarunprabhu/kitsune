; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --check-prefix STRING
;
; STRING: expected integer

attributes #2 = { kit_tt(\"cuda\") }
