; Check that the command lines to external commands issued during fat binary
; generation are as expected.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:           -passes='kit-cgfb' -cgfb-### -disable-output 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: ptxas
; CHECK-SAME: --gpu-name sm_86
; CHECK-SAME: --warn-on-spills
; CHECK-SAME: --opt-level 1
; CHECK-SAME: --output-file [[ASMFILE:.+[.]s]]
; CHECK-SAME: {{.+}}.ptx
;
; CHECK: fatbinary
; CHECK-SAME: --64
; CHECK-SAME: --create
; CHECK-SAME: {{.+}}.cufatbin
