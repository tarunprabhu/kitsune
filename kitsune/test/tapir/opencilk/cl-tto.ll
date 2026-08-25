; Check that the tapir target options specific to the opencilk tapir target are
; set correctly.
;
; RUN: opt --tapir=opencilk %s -disable-output \
; RUN:     --tapir-opencilk-runtime-bc=%S/input/libopencilk-abi.bc \
; RUN:     -passes="kit-print-tt-options" 2>&1 \
; RUN:     | FileCheck %s -check-prefixes ALL,CHECK
;
; ALL:    Tapir target options
; ALL:    Primary: opencilk
; CHECK:  Opencilk bitcode file: {{.+}}/libopencilk-abi.bc
