; REQUIRES: kitsune-hip
;
; When passing --tapir=hip to opt directly, some options are required. Check
; that an appropriate error is emitted when these options are not provided.
; These options are only required when the 'tapir-lowering' or 'kit-lowering'
; meta-passes are specified.
;
; RUN: not opt --tapir=hip \
; RUN:     -passes='tapir-lowering<O1>' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=ARCH
;
; RUN: not opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:     -passes='tapir-lowering<O1>' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=RUNTIME-BCS
;
; ARCH: error: option '--tapir-hip-arch' must be provided exactly once
; RUNTIME-BCS: error: option '--tapir-hip-runtime-bcs' must be provided exactly once
;
; ------------------------------------------------------------------------------
; This runs the loop-spawning pass which does use the tapir target options
; object. However, here, the fact that the object does not have all the
; "required" options set does not have any negative effects.
;
; RUN: opt --tapir=hip -passes='loop-spawning' %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --allow-empty --check-prefix=NOOUT
;
; NOOUT-NOT: {{^.+$}}
