; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

@gc1 = constant [0 x i8] zeroinitializer, !kitsune.fb !0
@gc2 = constant [0 x i8] zeroinitializer, !kitsune.fb !0
@.g2 = constant [0 x i8] zeroinitializer, !kitsune.fb !1

!0 = !{i8 2}
!1 = !{i8 3}

; CHECK: too many embedded fat binary globals for tapir target 'cuda'