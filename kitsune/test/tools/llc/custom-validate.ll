; Check that the values of the command line options used by the custom tapir
; target are validated.
;
; RUN: not llc --tapir=custom --tapir-plugin= \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=MISSING
;
; MISSING: error: for the --tapir-plugin option: value '' is invalid
;
; RUN: not llc --tapir=custom --tapir-plugin=noexist \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=NOEXIST
;
; NOEXIST: error: Could not load library
; NOEXIST-SAME: cannot open shared object file
; NOEXIST-SAME: No such file or directory
;
; RUN: not llc --tapir=custom --tapir-plugin=%s \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CONTENTS
;
; CONTENTS: error: Could not load library
