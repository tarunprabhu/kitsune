; Check that valid cgfb optimization levels are handled correctly.
;
; ------------------------------------------------------------------------------
; If a -cgfb-O<N> option is not provided, use the optimization level from the
; main tapir target options.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-hip-arch=gfx90a --tapir-lld=ld.lld -S \
; RUN:           --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:           -passes='tapir-lowering<O1>,kit-cgfb' -disable-output \
; RUN:           -cgfb-debug-target-machine 2>&1 \
; RUN:     | FileCheck %s --check-prefix O1
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-hip-arch=gfx90a --tapir-lld=ld.lld -S \
; RUN:           --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:           -passes='tapir-lowering<O3>,kit-cgfb' -disable-output \
; RUN:           -cgfb-debug-target-machine 2>&1 \
; RUN:     | FileCheck %s --check-prefix O3
;
; ------------------------------------------------------------------------------
; Otherwise, check that the optimization level makes it to the target machine.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-lld=ld.lld -disable-output \
; RUN:           -passes='kit-cgfb' -cgfb-O0 -cgfb-debug-target-machine 2>&1 \
; RUN:     | FileCheck %s --check-prefix O0
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-lld=ld.lld -disable-output \
; RUN:           -passes='kit-cgfb' -cgfb-O1 -cgfb-debug-target-machine 2>&1 \
; RUN:     | FileCheck %s --check-prefix O1
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-lld=ld.lld -disable-output \
; RUN:           -passes='kit-cgfb' -cgfb-O2 -cgfb-debug-target-machine 2>&1 \
; RUN:     | FileCheck %s --check-prefix O2
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-lld=ld.lld -disable-output \
; RUN:           -passes='kit-cgfb' -cgfb-O3 -cgfb-debug-target-machine 2>&1 \
; RUN:     | FileCheck %s --check-prefix O3
;
; ------------------------------------------------------------------------------
;
; O0: Optimization level: none (O0)
; O1: Optimization level: less (O1)
; O2: Optimization level: default (O2)
; O3: Optimization level: aggressive (O3)
;
; ------------------------------------------------------------------------------
