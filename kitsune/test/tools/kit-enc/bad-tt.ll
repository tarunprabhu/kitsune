; Check that the correct error message is shown when a tapir target that does
; not generate embedded bitcode is specified.
;
; RUN: not %kit-enc -tapir=serial %s 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: 'serial' tapir target does not generate embedded bitcode
