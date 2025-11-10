; Check that the code model in the target machine is set correctly when
; generating the fat binary.
;
; At this time, we always use the small code model. It is unlikely that we will
; ever use anything else, or make this configurable.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt -o /dev/null --tapir=hip --tapir-hip-arch=gfx906 \
; RUN:           --tapir-lld=ld.lld \
; RUN:           --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:           -passes='tapir-lowering<O1>,kit-cgfb' \
; RUN:           -cgfb-debug-target-machine 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: Code model: small
