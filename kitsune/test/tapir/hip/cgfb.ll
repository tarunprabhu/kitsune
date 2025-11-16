; Check that the fat binary is generated correctly. The global containing
; embedded bitcode should be removed after the fat binary is generated.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-hip-arch=gfx90a --tapir-lld=ld.lld -S \
; RUN:           --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:           -passes='kit-cgfb' \
; RUN:     | FileCheck %s
;
; CHECK-NOT: @{{.+}}.bc = constant [{{[0-9]+}} x i8] c"{{.+}}"
; CHECK: @{{.+}}.fb = constant [{{[0-9]+}} x i8] c"{{[^,]+}}",
; CHECK-SAME: section ".hip_fatbin"
; CHECK-SAME: #[[FBATTR:[0-9]+]]
;
; CHECK: attributes #[[FBATTR]] = { kit_fb kit_tt(4) }
; CHECK-NOT: #{{[0-9]+}} = {{.+}} kit_bc kit_tt(4)
