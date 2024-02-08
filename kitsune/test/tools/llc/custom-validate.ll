; Check that the values of the command line options used by the custom tapir
; target are validated.
;
; RUN: not llc --tapir=custom --tapir-plugin= \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=MISSING
;
; MISSING: error: option '--tapir-plugin' has invalid value ''
;
; RUN: not llc --tapir=custom --tapir-plugin=noexist \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=NOEXIST
;
; The actual error message for a non-existent file is system-dependent. While
; it would be nice if we could check that the correct error message is shown
; in such cases, it is probably not worth the extra effort.
;
; NOEXIST: error: could not load plugin library
;
; RUN: not llc --tapir=custom --tapir-plugin=%s \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CONTENTS
;
; CONTENTS: error: could not load plugin library
