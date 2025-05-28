; If embedded bitcode is present, a corresponding fat binary global must be
; present too.
;
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

@0 = constant [0 x i8] zeroinitializer, !kitsune.bc !0

!0 = !{i8 2}

; CHECK: embedded bitcode global without fat binary global