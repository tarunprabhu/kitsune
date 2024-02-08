; Check that sramecc is handled correctly when generating the fat binary
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-lld=ld.lld --tapir-hip-sramecc=any \
; RUN:           -passes='kit-cgfb' -cgfb-### -disable-output \
; RUN:     | FileCheck %s --check-prefixes ALL,ANY
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-lld=ld.lld --tapir-hip-sramecc=on \
; RUN:           -passes='kit-cgfb' -cgfb-### -disable-output \
; RUN:     | FileCheck %s --check-prefixes ALL,ON
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-lld=ld.lld --tapir-hip-sramecc=off \
; RUN:           -passes='kit-cgfb' -cgfb-### -disable-output \
; RUN:     | FileCheck %s --check-prefixes ALL,OFF
;
; ALL: --plugin-opt=-mcpu={{[^:]+}}
; ANY-NOT: :sramecc+
; ANY-NOT: :sramecc-
; ON-SAME: :sramecc+
; OFF-SAME: :sramecc-
