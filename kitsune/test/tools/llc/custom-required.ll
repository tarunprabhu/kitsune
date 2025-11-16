; When passing --tapir=custom to llc directly, some options are required. Check
; that an appropriate error is emitted when these options are not provided.
;
; RUN: not llc --tapir=custom -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=PLUGIN
;
; PLUGIN: error: the --tapir-plugin option must be provided exactly once
