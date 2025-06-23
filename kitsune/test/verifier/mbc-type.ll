; Global variables with the kit_bc attribute must have type [n x i8] where n can
; be any positive integer.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: incorrect type of global containing bitcode

@bc = constant i64 11 #0
@fb = constant [0 x i8] zeroinitializer #1

attributes #0 = { kit_bc(8) }
attributes #1 = { kit_fb(8) }
