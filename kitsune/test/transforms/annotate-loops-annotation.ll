; Check that the kit-annotate-prelower pass adds the correct annotation after
; it is run, even if it does nothing.
;
; RUN: opt -passes='kit-annotate-prelower' -S %s \
; RUN:     | FileCheck %s
;
; CHECK: !kit.module = !{![[MD:[0-9]+]]}
;
; CHECK-DAG: ![[MD]] = distinct !{![[MD]], ![[FLAG:[0-9]+]]}
; CHECK-DAG: ![[FLAG]] = !{!"kit.module.pre.lower.annotate.pass"}
