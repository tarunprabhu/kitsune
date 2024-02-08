; Check that the tapir target options specific to the serial tapir target are
; set correctly depending on the corresponding options passed to LLVM's opt
; utility.
;
; Currently, there are no options specific to the serial tapir target. We just
; check that the tapir target ID is set correctly.
;
; RUN: opt --tapir=serial %s -disable-output \
; RUN:     -passes="loop-spawning" -dump-tapir-target-options \
; RUN:     | FileCheck %s -check-prefixes ALL
;
; ALL: Tapir target options
; ALL: Primary: serial
