; Check that the values of the command line options used by the custom tapir
; target are validated by opt.
;
; RUN: not opt --tapir=custom --tapir-plugin= \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=MISSING
;
; MISSING: error: option '--tapir-plugin' has invalid value ''
;
; RUN: not opt --tapir=custom --tapir-plugin=noexist \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=NOEXIST
;
; The actual error message for a non-existent file is system-dependent. While
; it would be nice if we could check that the correct error message is shown
; in such cases, it is probably not worth the extra effort.
;
; NOEXIST: error: could not load plugin library
;
; RUN: not opt --tapir=custom --tapir-plugin=%s \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CONTENTS
;
; CONTENTS: error: could not load plugin library
