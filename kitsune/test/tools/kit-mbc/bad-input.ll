; If the input module could not be parsed, check that the error is as expected
;
; RUN: not %kit-mbc %s 2>&1 | FileCheck %s
; RUN: echo "BC" | not %kit-mbc 2>&1 | FileCheck %s
;
; CHECK: error:

BC
