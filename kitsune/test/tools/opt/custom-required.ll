; When passing --tapir=custom to opt directly, some options are required. Check
; that an appropriate error is emitted when these options are not provided.
; These options are only required when the 'tapir-lowering' or 'kit-lowering'
; meta-passes are specified.
;
; RUN: not opt --tapir=custom \
; RUN:     -passes='tapir-lowering<O1>' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=PLUGIN
;
; PLUGIN: error: the --tapir-plugin option must be provided exactly once
;
; ------------------------------------------------------------------------------
; This runs the loop-vectorize pass instead of loop-spawning. loop-spawning
; will attempt to run the tapir target constructor in the plugin which will,
; obviously, result in a crash since the plugin will not exist. This behavior is
; intentional; we want users to have the ability to use potentially unsafe
; combinations of options, but we do provide some error checking for use cases
; that are very likely to result in failures.
;
; RUN: opt --tapir=custom -passes='loop-vectorize' %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --allow-empty --check-prefix=NOOUT
;
; NOOUT-NOT: {{^.+$}}
