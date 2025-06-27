; RUN: not llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --check-prefix UNKNOWN
;
; UNKNOWN: unknown tapir target

attributes #3 = { kit_tt(20834289) }
