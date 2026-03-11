; Check that the relocation model in the target machine is set correctly when
; generating the fat binary.
;
; Currently, this always uses PIC as the relocation model. If that is ever made
; configurable, or if the model in the target machine is derived from the
; TTOptions, this should be changed.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt -o /dev/null --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:           --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:           -passes='kit-cgfb' \
; RUN:           -cgfb-debug-target-machine 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: Relocation model: pic
