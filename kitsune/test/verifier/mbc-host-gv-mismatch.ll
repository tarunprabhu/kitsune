; The kitsune device metadata in the embedded bitcode contains the tapir
; target that generated the module. This must match the tapir target set on
; the global variable in the host that contains the bitcode.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | sed 's/kit_bc kit_tt(2)/kit_bc kit_tt(4)/g' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: tapir target in embedded module must match tapir target in host embedded bitcode global variable
