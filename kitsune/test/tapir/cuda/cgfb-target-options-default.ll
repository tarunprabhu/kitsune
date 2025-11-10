; When creating the target machine to be used for code generation, some target
; options are set explicitly. Check that these options are as expected.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt -o /dev/null --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:           --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:           -passes='tapir-lowering<O1>,kit-cgfb' \
; RUN:           -cgfb-debug-target-options 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: AllowFPOpFusion: standard
