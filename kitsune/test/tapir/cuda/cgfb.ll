; Check that the fat binary is generated correctly. The global containing
; embedded bitcode should be removed after the fat binary is generated.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt --tapir=cuda -passes='kit-cgfb' -S \
; RUN:     | FileCheck %s
;
; CHECK-NOT: @{{.+}}.bc{{.*}} = constant [{{[0-9]+}} x i8] c"{{.+}}"
; CHECK: @{{.+}}.fb{{.*}} = constant [{{[0-9]+}} x i8] c"{{[^,]+}}",
; CHECK-SAME: section ".nv_fatbin"
; CHECK-SAME: !kit.gv ![[MD:[0-9]+]]
;
; CHECK-DAG: ![[MD]] = distinct !{![[MD]], ![[DC:[0-9]+]]}
; CHECK-DAG: ![[DC]] = !{!"kit.gv.device.code", i32 2}
