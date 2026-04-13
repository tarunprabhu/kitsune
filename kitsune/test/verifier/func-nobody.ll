; The implementation of Kitsune's function verifier creates a dominator tree and
; loop info object. Doing this on a function without a body results in a crash.
; This test ensures that the implementation does not attempt to create a
; dominator tree for a function without a body.
;
; RUN: llvm-as %s -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{.+}}

declare void @f()
