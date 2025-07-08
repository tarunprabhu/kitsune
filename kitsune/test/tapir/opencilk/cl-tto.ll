; Check that opt's command line options make it to the tapir target options.
;
; RUN: opt --tapir=opencilk -passes="tapir-lowering<O2>" -o /dev/null %s \
; RUN:     -dump-tapir-target-options \
; RUN:     --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc 2>&1 \
; RUN:     | FileCheck %s -check-prefixes ALL,CHECK
;
; ALL:    Tapir target options
; ALL:    Primary: opencilk
; CHECK:  Opencilk bitcode file: {{.+}}/libopencilk-abi.bc
