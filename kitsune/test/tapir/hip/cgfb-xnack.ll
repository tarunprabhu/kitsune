; Check that xnack is handled correctly when generating the fat binary
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-hip-arch=gfx90a --tapir-lld=ld.lld -S \
; RUN:           --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:           --tapir-hip-xnack=any -cgfb-### \
; RUN:           -passes='kit-cgfb' \
; RUN:     | FileCheck %s --check-prefixes ALL,ANY
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-hip-arch=gfx90a --tapir-lld=ld.lld -S \
; RUN:           --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:           --tapir-hip-xnack=on -cgfb-### \
; RUN:           -passes='kit-cgfb' \
; RUN:     | FileCheck %s --check-prefixes ALL,ON
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-hip-arch=gfx90a --tapir-lld=ld.lld -S \
; RUN:           --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:           --tapir-hip-xnack=off -cgfb-### \
; RUN:           -passes='kit-cgfb' \
; RUN:     | FileCheck %s --check-prefixes ALL,OFF
;
; ALL: --plugin-opt=-mcpu=gfx90a
; ANY-NOT: :xnack+
; ANY-NOT: :xnack-
; ON-SAME: :xnack+
; OFF-SAME: :xnack-
