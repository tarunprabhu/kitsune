; The final reduction value must be loaded from an alloca.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | not opt -passes='emb-finalize-reductions' 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: value of final reduction must be loaded from an alloca

define void @f(ptr %res, i64 %v) !kit.func !0 {
  call void (i32, i32, ptr, i64, i32, i32, i64, ptr, ...) @llvm.kit.reduce(i32 1024, i32 5, ptr %res, i64 %v, i32 10, i32 8, i64 0, ptr @sum)
  ret void
}

declare void @sum(ptr, i64)

!0 = distinct !{!0, !1}
!1 = !{!"kit.func.kernel", i32 1}
