; The kernel function must contain exactly one top-level loop.
;
; RUN: %kit-enc %s \
; RUN:     | not opt -passes='emb-finalize-reductions' 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: kernel function must contain exactly one top-level loop

define void @f() !kit.func !0 {
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"kit.func.kernel", i32 1}
