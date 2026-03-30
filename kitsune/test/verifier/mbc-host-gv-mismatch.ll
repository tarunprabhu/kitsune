; The kitsune device metadata in the embedded bitcode contains the tapir
; target that generated the module. This must match the tapir target set on
; the global variable in the host that contains the bitcode.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | sed 's/i32 2/i32 4/g' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | sed 's/i32 4/i32 2/g' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: tapir target in device module flags metadata must match tapir target in host embedded bitcode global variable
