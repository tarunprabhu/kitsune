; REQUIRES: kitsune-opencilk
;
; When passing --tapir=opencilk to opt directly, some options are required.
; Check that an appropriate error is emitted when these options are not
; provided. These options are only required when the 'tapir-lowering' or
; 'kit-lowering' meta-passes are specified.
;
; RUN: not opt --tapir=opencilk %s -disable-output \
; RUN:     -passes='tapir-lowering<O1>' 2>&1 \
; RUN:     | FileCheck %s --check-prefix=RUNTIME-BC
;
; RUNTIME-BC: error: option '--tapir-opencilk-runtime-bc' must be provided exactly once
;
; ------------------------------------------------------------------------------
; This runs the loop-spawning pass which does use the tapir target options
; object. However, here, the fact that the object does not have all the
; "required" options set does not have any negative effects.
;
; RUN: opt --tapir=opencilk %s -disable-output \
; RUN:     -passes='loop-spawning' 2>&1 \
; RUN:     | FileCheck %s --allow-empty --check-prefix=NOOUT
;
; NOOUT-NOT: {{^.+$}}
