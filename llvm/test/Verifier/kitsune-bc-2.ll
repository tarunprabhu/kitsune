; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

@fb1 = constant [0 x i8] zeroinitializer, !kitsune.fb !0
@fb2 = constant [0 x i8] zeroinitializer, !kitsune.fb !1
@gc1 = constant [0 x i8] zeroinitializer, !kitsune.bc !0
@gc2 = constant [0 x i8] zeroinitializer, !kitsune.bc !1
@.g2 = constant [0 x i8] zeroinitializer, !kitsune.bc !1

!0 = !{i8 2}
!1 = !{i8 4}

; CHECK: too many embedded bitcode globals for tapir target 'hip'
