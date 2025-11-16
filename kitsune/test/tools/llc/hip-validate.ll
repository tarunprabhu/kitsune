; REQUIRES: kitsune-hip
;
; Check that the values of the command line options used by the hip tapir
; target are validated when passed to llc.
;
; ------------------------------------------------------------------------------
; RUN: not llc --tapir=hip --tapir-hip-arch= \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice-amd.ll \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=ARCH
;
; ARCH: error: for the --tapir-hip-arch option: value '' is invalid
;
; ------------------------------------------------------------------------------
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs= \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC
;
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=,,, \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC
;
; BC: error: for the --tapir-hip-runtime-bcs option:
; BC-SAME: at least one valid value is required
;
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=noexist \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC-NOEXIST
;
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice-amd.ll,noexist \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC-NOEXIST
;
; BC-NOEXIST: error: failed to parse LLVM file
; BC-NOEXIST-SAME: Could not open input file
; BC-NOEXIST-SAME: No such file or directory
;
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/bogus.ll \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC-CONTENTS
;
; BC-CONTENTS: error: failed to parse LLVM file
; BC-CONTENTS-SAME: expected top-level entity
;
; ------------------------------------------------------------------------------
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice-amd.ll \
; RUN:         --tapir-hip-features= \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=FEATURES
;
; FEATURES: error: for the --tapir-hip-features option: value '' is invalid
;
; ------------------------------------------------------------------------------
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice-amd.ll \
; RUN:         --tapir-hip-sramecc= \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=SRAMECC
;
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice-amd.ll \
; RUN:         --tapir-hip-sramecc=1 \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=SRAMECC
;
; SRAMECC: for the --tapir-hip-sramecc option:
; SRAMECC-SAME: Cannot find option named '{{.*}}'
;
; ------------------------------------------------------------------------------
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice-amd.ll \
; RUN:         --tapir-hip-xnack= \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=XNACK
;
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice-amd.ll \
; RUN:         --tapir-hip-xnack=1 \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=XNACK
;
; XNACK: for the --tapir-hip-xnack option:
; XNACK-SAME: Cannot find option named '{{.*}}'
;
; ------------------------------------------------------------------------------
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice-amd.ll \
; RUN:         --tapir-gpu-tpb= \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=TPB
;
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice-amd.ll \
; RUN:         --tapir-gpu-tpb=ten \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=TPB
;
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice-amd.ll \
; RUN:         --tapir-gpu-tpb=0 \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=RANGE
;
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice-amd.ll \
; RUN:         --tapir-gpu-tpb=1025 \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=RANGE
;
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice-amd.ll \
; RUN:         --tapir-gpu-max-tpb= \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=TPB
;
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice-amd.ll \
; RUN:         --tapir-gpu-max-tpb=0 \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=RANGE
;
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice-amd.ll \
; RUN:         --tapir-gpu-max-tpb=1025 \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=RANGE
;
; TPB: for the --tapir-gpu{{(-.+)?}}-tpb option: '{{.*}}' value invalid
; RANGE: error: for the --tapir-gpu{{(-.+)?}}-tpb option:
; RANGE-SAME: value '{{.+}}' is not in range [1,1024]
;
; ------------------------------------------------------------------------------
