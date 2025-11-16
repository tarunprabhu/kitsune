; REQUIRES: kitsune-opencilk
;
; Check that the values of the command line options used by the opencilk tapir
; target are validated when passed to llc.
;
; RUN: not llc --tapir=opencilk \
; RUN:         --tapir-opencilk-runtime-bc= \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC
;
; BC: error: for the --tapir-opencilk-runtime-bc option: value '' is invalid
;
; RUN: not llc --tapir=opencilk \
; RUN:         --tapir-opencilk-runtime-bc=noexist \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC-NOEXIST
;
; BC-NOEXIST: error: failed to parse LLVM file
; BC-NOEXIST-SAME: Could not open input file
; BC-NOEXIST-SAME: No such file or directory
;
; RUN: not llc --tapir=opencilk \
; RUN:         --tapir-opencilk-runtime-bc=%S/input/bogus.ll \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC-CONTENTS
;
; BC-CONTENTS: error: failed to parse LLVM file
; BC-CONTENTS-SAME: expected top-level entity
