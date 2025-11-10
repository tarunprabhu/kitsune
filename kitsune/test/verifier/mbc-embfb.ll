; The embedded bitcode in this file contains a fat binary global variable. This
; is not allowed.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: embedded module cannot contain an embedded fat binary

@fb = constant [0 x i8] zeroinitializer #0

attributes #0 = { kit_fb kit_tt(2) }
