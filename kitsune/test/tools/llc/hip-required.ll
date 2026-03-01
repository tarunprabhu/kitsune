; REQUIRES: kitsune-hip
;
; When passing --tapir=hip to llc directly, some options are required. Check
; that an appropriate error is emitted when these options are not provided.
;
; RUN: not llc --tapir=hip -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=ARCH
;
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c -filetype=asm -o /dev/null \
; RUN:     %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=RUNTIME-BCS
;
; ARCH: error: option '--tapir-hip-arch' must be provided exactly once
; RUNTIME-BCS: error: option '--tapir-hip-runtime-bcs' must be provided exactly once
