; If embedded bitcode is present, a corresponding fat binary global must be
; present too.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: embedded bitcode global without fat binary global

@0 = constant [0 x i8] zeroinitializer #0

attributes #0 = { kit_bc kit_tt(1) }
