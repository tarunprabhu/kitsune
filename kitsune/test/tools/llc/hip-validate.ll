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
; ARCH: error: option '--tapir-hip-arch' has invalid value ''
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
; BC: error: option '--tapir-hip-runtime-bcs' requires at least one valid value
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
; BC-NOEXIST: error: could not parse LLVM file
; BC-NOEXIST-SAME: Could not open input file
; BC-NOEXIST-SAME: No such file or directory
;
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/bogus.ll \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC-CONTENTS
;
; BC-CONTENTS: error: could not parse LLVM file
; BC-CONTENTS-SAME: expected top-level entity
;
; ------------------------------------------------------------------------------
; RUN: not llc --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice-amd.ll \
; RUN:         --tapir-hip-features= \
; RUN:         -filetype=asm -o /dev/null %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=FEATURES
;
; FEATURES: error: option '--tapir-hip-features' has invalid value ''
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
