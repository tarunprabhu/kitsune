; Check that the kit-serialize pass adds the correct annotation after it is run,
; even if it does nothing.
;
; RUN: opt -passes='kit-serialize' -S %s \
; RUN:     | FileCheck %s
;
; CHECK: !kit.module = !{![[MD:[0-9]+]]}
;
; CHECK-DAG: ![[MD]] = distinct !{![[MD]], ![[ANNOTATE:[0-9]+]], ![[SERIALIZE:[0-9]+]]}
; CHECK-DAG: ![[ANNOTATE]] = !{!"kit.module.pre.lower.annotate.pass"}
; CHECK-DAG: ![[SERIALIZE]] = !{!"kit.module.serialize.pass"}

!kit.module = !{!0}

!0 = distinct !{!0, !1}
!1 = !{!"kit.module.pre.lower.annotate.pass"}
