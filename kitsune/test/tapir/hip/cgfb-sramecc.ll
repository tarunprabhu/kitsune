; Check that sramecc is handled correctly when generating the fat binary
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-hip-arch=gfx90a --tapir-lld=ld.lld -S \
; RUN:           --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:           --tapir-hip-sramecc=any -cgfb-### \
; RUN:           -passes='kit-cgfb' -o /dev/null \
; RUN:     | FileCheck %s --check-prefixes ALL,ANY
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-hip-arch=gfx90a --tapir-lld=ld.lld -S \
; RUN:           --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:           --tapir-hip-sramecc=on -cgfb-### \
; RUN:           -passes='kit-cgfb' -o /dev/null \
; RUN:     | FileCheck %s --check-prefixes ALL,ON
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-hip-arch=gfx90a --tapir-lld=ld.lld -S \
; RUN:           --tapir-hip-runtime-bcs=%S/input/libdevice.ll \
; RUN:           --tapir-hip-sramecc=off -cgfb-### \
; RUN:           -passes='kit-cgfb' -o /dev/null \
; RUN:     | FileCheck %s --check-prefixes ALL,OFF
;
; ALL: --plugin-opt=-mcpu=gfx90a
; ANY-NOT: :sramecc+
; ANY-NOT: :sramecc-
; ON-SAME: :sramecc+
; OFF-SAME: :sramecc-
