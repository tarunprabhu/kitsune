; Check that invalid cgfb optimization levels are handled correctly.
;
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_80 \
; RUN:     -passes='kit-cgfb' -cgfb-Os \
; RUN:     -o /dev/null %S/input/empty.ll 2>&1 \
; RUN:     | FileCheck %s --check-prefix OS
;
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_80 \
; RUN:     -passes='kit-cgfb' -cgfb-Oz \
; RUN:     -o /dev/null %S/input/empty.ll 2>&1 \
; RUN:     | FileCheck %s --check-prefix OZ
;
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_80 \
; RUN:     -passes='kit-cgfb' -cgfb-O4 \
; RUN:     -o /dev/null %S/input/empty.ll 2>&1 \
; RUN:     | FileCheck %s --check-prefix O4
;
; OS: Unknown command line argument '-cgfb-Os'
; OZ: Unknown command line argument '-cgfb-Oz'
; O4: Unknown command line argument '-cgfb-O4'
