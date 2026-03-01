; REQUIRES: kitsune-opencilk
;
; When passing --tapir=opencilk to llc directly, some options are required.
; Check that an appropriate error is emitted when these options are not
; provided.
;
; RUN: not llc --tapir=opencilk -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=RUNTIME-BC
;
; RUNTIME-BC: error: option '--tapir-opencilk-runtime-bc' must be provided exactly once
