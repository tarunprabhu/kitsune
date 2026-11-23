; The destination of a warp shuffle reduction must have a single use.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | not opt -passes='emb-finalize-reductions' 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: incorrect number of uses. Expected 1. Got 2

define void @f(ptr %res) !kit.func !0 {
  %l1 = alloca i64
  %l2 = alloca i64
  %v1 = load i64, ptr %l1
  %v2 = load i64, ptr %l2
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1024, i32 5, ptr %res, i64 %v1, i32 14, i32 8, i64 0, ptr @sum)
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1024, i32 5, ptr %res, i64 %v2, i32 14, i32 8, i64 0, ptr @sum)
  ret void
}

declare void @sum(ptr, i64)

!0 = distinct !{!0, !1}
!1 = !{!"kit.func.kernel", i32 1}
