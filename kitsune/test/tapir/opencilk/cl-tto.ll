; Check that opt's command line options make it to the tapir target options.
;
; RUN: opt --tapir=opencilk %s -disable-output \
; RUN:     --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
; RUN:     -passes="loop-spawning" -dump-tapir-target-options 2>&1 \
; RUN:     | FileCheck %s -check-prefixes ALL,CHECK
;
; ALL:    Tapir target options
; ALL:    Primary: opencilk
; CHECK:  Opencilk bitcode file: {{.+}}/libopencilk-abi.bc
