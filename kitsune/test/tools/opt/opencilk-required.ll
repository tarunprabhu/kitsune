; REQUIRES: kitsune-opencilk
;
; When passing --tapir=opencilk to opt directly, some options are required.
; Check that an appropriate error is emitted when these options are not
; provided. These options are only required when the 'tapir-lowering' or
; 'kit-lowering' meta-passes are specified.
;
; RUN: not opt --tapir=opencilk \
; RUN:     -passes='tapir-lowering<O1>' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=RUNTIME-BC
;
; RUNTIME-BC: error: the --tapir-opencilk-runtime-bc option must be provided exactly once
;
; ------------------------------------------------------------------------------
; This runs the loop-spawning pass which does use the tapir target options
; object. However, here, the fact that the object does not have all the
; "required" options set does not have any negative effects.
;
; RUN: opt --tapir=opencilk -passes='loop-spawning' %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --allow-empty --check-prefix=NOOUT
;
; NOOUT-NOT: {{^.+$}}
