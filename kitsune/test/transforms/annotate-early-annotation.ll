; kit-annotate-early is a requirable psas. It should add an annotation
; indicating that it has been run on a function, even if it did not do add any
; annotations to the function.
;
; RUN: opt -passes='kit-annotate-early' -S %s 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: @f
; CHECK-SAME: !kit.func ![[ATTRS:[0-9]+]]
;
; CHECK-LABEL: @g
; CHECK-NOT: !kit.func !{{[0-9]+}}
;
; CHECK-DAG: ![[ATTR:[0-9]+]] = !{!"kit.func.early.annotate.pass"}
; CHECK-DAG: ![[ATTRS]] = distinct !{![[ATTRS]], ![[ATTR]]}

define void @f() {
  call void @g()
  ret void
}

declare void @g()
