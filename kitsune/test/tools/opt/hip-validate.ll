; REQUIRES: kitsune-hip
;
; Check that the values of the command line options used by the hip tapir
; target are validated by opt.
;
; ------------------------------------------------------------------------------
; RUN: not opt --tapir=hip --tapir-hip-arch= \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=ARCH
;
; ARCH: error: option '--tapir-hip-arch' has invalid value ''
;
; ------------------------------------------------------------------------------
; RUN: not opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs= \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC
;
; RUN: not opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=,,, \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC
;
; BC: error: option '--tapir-hip-runtime-bcs' requires at least one valid value
;
; RUN: not opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=noexist \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC-NOEXIST
;
; RUN: not opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/amd.bc,noexist,%S/input/libdevice.ll \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC-NOEXIST
;
; BC-NOEXIST: error: could not parse LLVM file
; BC-NOEXIST-SAME: Could not open input file
; BC-NOEXIST-SAME: No such file or directory
;
; RUN: not opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/bogus.ll \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=BC-CONTENTS
;
; BC-CONTENTS: error: could not parse LLVM file
; BC-CONTENTS-SAME: expected top-level entity
;
; ------------------------------------------------------------------------------
; RUN: not opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:         --tapir-hip-features= \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=FEATURES
;
; FEATURES: error: option '--tapir-hip-features' has invalid value ''
;
; ------------------------------------------------------------------------------
; RUN: not opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:         --tapir-hip-sramecc= \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=SRAMECC
;
; RUN: not opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:         --tapir-hip-sramecc=1 \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=SRAMECC
;
; SRAMECC: for the --tapir-hip-sramecc option:
; SRAMECC-SAME: Cannot find option named '{{.*}}'
;
; ------------------------------------------------------------------------------
; RUN: not opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:         --tapir-hip-xnack= \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=XNACK
;
; RUN: not opt --tapir=hip --tapir-hip-arch=gfx90c \
; RUN:         --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:         --tapir-hip-xnack=1 \
; RUN:         -passes='kit-lowering<O1>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=XNACK
;
; XNACK: for the --tapir-hip-xnack option:
; XNACK-SAME: Cannot find option named '{{.*}}'
;
; ------------------------------------------------------------------------------
