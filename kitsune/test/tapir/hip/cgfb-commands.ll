; Check that the command lines to external commands issued during fat binary
; generation are as expected.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-hip-arch=gfx906 --tapir-lld=ld.lld \
; RUN:           -passes='kit-cgfb' -cgfb-### -o /dev/null 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: lld{{[^ ]*}}
; CHECK-SAME: -flavor gnu
; CHECK-SAME: -m elf64_amdgpu
; CHECK-SAME: --no-undefined
; CHECK-SAME: -shared
; CHECK-SAME: --eh-frame-hdr
; CHECK-SAME: --plugin-opt=-amdgpu-internalize-symbols
; CHECK-SAME: --plugin-opt=-mcpu=gfx906
; CHECK-SAME: -o {{.+}}.so
; CHECK-SAME: {{.+}}.o
