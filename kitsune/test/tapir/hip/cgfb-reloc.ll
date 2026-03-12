; Check that the relocation model in the target machine is set correctly when
; generating the fat binary.
;
; At the time of writing, 28-Jun-2025, the AMDTargetMachine *requires* the
; relocation model to be PIC since the fat binary that it generates is a shared
; object. It ignores the Reloc::Model parameter that is provided to the
; constructor.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-lld=ld.lld \
; RUN:           -passes='kit-cgfb' -disable-output \
; RUN:           -cgfb-debug-target-machine 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: Relocation model: pic
