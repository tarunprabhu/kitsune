; Calls to Kitsune's reduce intrinsic with a warp shuffle strategy must only
; be present in kernel functions.
;
; RUN: %kit-enc %s \
; RUN:     | not opt -passes='emb-finalize-reductions' 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: reductions with the warp shuffle strategy are only allowed in kernel functions

define void @f(ptr %res, i64 %n) !kit.func !0 {
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1024, i32 5, ptr %res, i64 %n, i32 10, i32 8, i64 0, ptr @sum)
  ret void
}

declare void @sum(ptr, i64)

!0 = distinct !{!0, !1}
!1 = !{!"kit.func.device"}
