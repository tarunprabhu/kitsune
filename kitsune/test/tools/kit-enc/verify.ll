; Check that the module produced by the tool passes the verifier. If the
; verifier succeeds, nothing should be printed.
;
; RUN: %kit-enc %s \
; RUN:     | opt -passes=verify -disable-output 2>&1 \
; RUN:     | FileCheck %s --allow-empty
;
; CHECK-NOT: {{.+}}
