; Check that the tapir target options specific to the serial tapir target are
; set correctly.
;
; NOTE: Currently, there are no such options, so this is just a placeholder for
; consistency with the tests for the other tapir targets.
;
; RUN: opt --tapir=serial %s -disable-output \
; RUN:     -passes="kit-print-tt-options" \
; RUN:     | FileCheck %s -check-prefixes ALL
;
; ALL: Tapir target options
; ALL: Primary: serial
