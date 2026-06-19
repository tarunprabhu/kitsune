; Check that the kit-serialize pass adds the correct annotation after it is run,
; even if it does nothing.
;
; RUN: opt -passes='kit-serialize' -S %s \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: !kit.func ![[MD:[0-9]+]]
;
; CHECK-DAG: ![[MD]] = distinct !{![[MD]], ![[ANNOTATE:[0-9]+]], ![[SERIALIZE:[0-9]+]]}
; CHECK-DAG: ![[ANNOTATE]] = !{!"kit.func.pre.lower.annotate.pass"}
; CHECK-DAG: ![[SERIALIZE]] = !{!"kit.func.serialize.pass"}

define void @f() !kit.func !0 {
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"kit.func.pre.lower.annotate.pass"}
