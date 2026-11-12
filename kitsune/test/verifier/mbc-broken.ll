; Check that the verifier fails if at least one embedded bitcode module contains
; an error. In this case, the embedded bitcode contains an invalid global
; variable - one with common linkage and a non-zero initializer.
;
; NOTE: This is one of the ways in which we can construct an invalid module
; without the verifier failing. If this is ever fixed in LLVM, we may have to
; find a different way of doing this.
;
; ------------------------------------------------------------------------------
; RUN: echo "@0 = common global i32 11" \
; RUN:     | %kit-enc --tapir=cuda \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; ------------------------------------------------------------------------------
; RUN: echo "@0 = common global i32 11" \
; RUN:     | %kit-enc --tapir=hip \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; ------------------------------------------------------------------------------
; CHECK: 'common' global must have a zero initializer
; CHECK: ptr @0
; CHECK: embedded module: broken module found
