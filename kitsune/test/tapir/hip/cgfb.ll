; Check that the fat binary is generated correctly. The global containing
; embedded bitcode should be removed after the fat binary is generated.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-lld=ld.lld -passes='kit-cgfb' -S \
; RUN:     | FileCheck %s
;
; CHECK-NOT: @{{.+}}.bc{{.*}} = constant [{{[0-9]+}} x i8] c"{{.+}}"
; CHECK: @{{.+}}.fb{{.*}} = constant [{{[0-9]+}} x i8] c"{{[^,]+}}",
; CHECK-SAME: section ".hip_fatbin"
; CHECK-SAME: !kit.gv.device.code ![[TT:[0-9]+]]
;
; CHECK: ![[TT]] = !{i32 4}
