; If the input module could not be parsed, check that the error is as expected
;
; RUN: not %kitmbc %s 2>&1 | FileCheck %s
; RUN: echo "BC" | not %kitmbc 2>&1 | FileCheck %s
;
; CHECK: error:

BC
