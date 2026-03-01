; REQUIRES: kitsune-cuda
;
; When passing --tapir=cuda to llc directly, some options are required. Check
; that an appropriate error is emitted when these options are not provided.
;
; RUN: not llc --tapir=cuda -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=ARCH
;
; RUN: not llc --tapir=cuda --tapir-cuda-arch=sm_86 -filetype=asm -o /dev/null \
; RUN:     %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=RUNTIME-BC
;
; ARCH: error: option '--tapir-cuda-arch' must be provided exactly once
; RUNTIME-BC: error: option '--tapir-cuda-runtime-bc' must be provided exactly once
