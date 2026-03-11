; When creating the target machine to be used for code generation, some target
; options are set explicitly. Check that these options are as expected.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-hip-arch=gfx906 --tapir-lld=ld.lld -S \
; RUN:           --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:           -passes='kit-cgfb' \
; RUN:           -cgfb-debug-target-options 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: AllowFPOpFusion: standard
; CHECK: EmitAddrsig: true
; CHECK: UseInitArray: true
