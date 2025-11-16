; Check that the -cgfb-ptxas-O<N> option is handled correctly.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:           --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:           -passes='kit-cgfb' -cgfb-### -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,DEFAULT
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt --tapir=cuda --tapir-cuda-arch=sm_86 -cgfb-ptxas-O0 \
; RUN:           --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:           -passes='kit-cgfb' -cgfb-### -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,O0
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt --tapir=cuda --tapir-cuda-arch=sm_86 -cgfb-ptxas-O1 \
; RUN:           --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:           -passes='kit-cgfb' -cgfb-### -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,O1
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt --tapir=cuda --tapir-cuda-arch=sm_86 -cgfb-ptxas-O2 \
; RUN:           --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:           -passes='kit-cgfb' -cgfb-### -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,O2
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt --tapir=cuda --tapir-cuda-arch=sm_86 -cgfb-ptxas-O3 \
; RUN:           --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:           -passes='kit-cgfb' -cgfb-### -o /dev/null 2>&1 \
; RUN:     | FileCheck %s --check-prefixes ALL,O3
;
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_86 -cgfb-ptxas-Os \
; RUN:         --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:         -passes='kit-cgfb' -cgfb-### -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes OS
;
; RUN: not opt --tapir=cuda --tapir-cuda-arch=sm_86 -cgfb-ptxas-Oz \
; RUN:         --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:         -passes='kit-cgfb' -cgfb-### -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes OZ
;
; ALL: ptxas
; DEFAULT-SAME: --opt-level 1
; O0-SAME: --opt-level 0
; O1-SAME: --opt-level 1
; O2-SAME: --opt-level 2
; O3-SAME: --opt-level 3
; OS: Unknown command line argument '-cgfb-ptxas-Os'
; OZ: Unknown command line argument '-cgfb-ptxas-Oz'

