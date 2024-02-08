; If the input module could not be parsed, check that the error is as expected
;
; RUN: not %kit-enc %s 2>&1 | FileCheck %s
; RUN: echo "BC" | not %kit-enc 2>&1 | FileCheck %s
;
; CHECK: error:

BC
