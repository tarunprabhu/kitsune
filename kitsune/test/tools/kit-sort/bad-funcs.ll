; If the requested functions do not exist in the module, check that the error
; is as expected.
;
; ------------------------------------------------------------------------------
; RUN: not %kit-sort %s --funcs=noexist 2>&1 \
; RUN:     | FileCheck %s --check-prefix=ONE
;
; ONE: function 'noexist' not found
;
; ------------------------------------------------------------------------------
; RUN: not %kit-sort %s --funcs=no1,no2 2>&1 \
; RUN:     | FileCheck %s --check-prefix=TWO
;
; TWO: function 'no1' not found
; TWO-NEXT: function 'no2' not found
