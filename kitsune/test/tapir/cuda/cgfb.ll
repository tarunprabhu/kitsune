; Check that the fat binary is generated correctly. The global containing
; embedded bitcode should be removed after the fat binary is generated.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt --tapir=cuda --tapir-cuda-arch=sm_80 -S \
; RUN:           -passes='kit-cgfb' \
; RUN:     | FileCheck %s
;
; CHECK-NOT: @{{.+}}.bc{{.*}} = constant [{{[0-9]+}} x i8] c"{{.+}}"
; CHECK: @{{.+}}.fb{{.*}} = constant [{{[0-9]+}} x i8] c"{{[^,]+}}",
; CHECK-SAME: section ".nv_fatbin"
; CHECK-SAME: #[[FBATTR:[0-9]+]]
;
; CHECK: #[[FBATTR]] = { kit_fb kit_tt(2) }
; CHECK-NOT: #{{[0-9]+}} = {{.+}} kit_bc kit_tt(2)
