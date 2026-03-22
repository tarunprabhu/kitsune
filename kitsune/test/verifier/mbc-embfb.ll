; The embedded bitcode in this file contains a global variable contain device
; code. This is not allowed.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: embedded module cannot contain embedded device code

@fb = constant [0 x i8] zeroinitializer, !kit.gv.device.code !0

!0 = !{i32 2}
