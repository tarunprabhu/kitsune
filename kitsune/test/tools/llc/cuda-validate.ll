; REQUIRES: kitsune-cuda
;
; Check that the values of the command line options used by the cuda tapir
; target are validated when passed to llc.
;
; ------------------------------------------------------------------------------
; RUN: not llc --tapir=cuda --tapir-cuda-arch= \
; RUN:         --tapir-cuda-runtime-bc=%S/input/libdevice-nv.ll \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=ARCH
;
; ARCH: error: option '--tapir-cuda-arch' has invalid value ''
;
; ------------------------------------------------------------------------------
; RUN: not llc --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc= \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC
;
; BC: error: option '--tapir-cuda-runtime-bc' has invalid value ''
;
; RUN: not llc --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc=noexist \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC-NOEXIST
;
; BC-NOEXIST: error: could not parse LLVM file
; BC-NOEXIST-SAME: Could not open input file
; BC-NOEXIST-SAME: No such file or directory
;
; RUN: not llc --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc=%S/input/bogus.ll \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC-CONTENTS
;
; BC-CONTENTS: error: could not parse LLVM file
; BC-CONTENTS-SAME: expected top-level entity
;
; ------------------------------------------------------------------------------
; RUN: not llc --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc=%S/input/libdevice-nv.ll \
; RUN:         --tapir-cuda-virt-arch= \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=VIRTARCH
;
; VIRTARCH: error: option '--tapir-cuda-virt-arch' has invalid value ''
;
; ------------------------------------------------------------------------------
; RUN: not llc --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:         --tapir-cuda-runtime-bc=%S/input/libdevice-nv.ll \
; RUN:         --tapir-cuda-features= \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=FEATURES
;
; FEATURES: error: option '--tapir-cuda-features' has invalid value ''
;
; ------------------------------------------------------------------------------
