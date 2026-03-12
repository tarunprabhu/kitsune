; Check that xnack is handled correctly when generating the fat binary
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-lld=ld.lld --tapir-hip-xnack=any \
; RUN:           -passes='kit-cgfb' -cgfb-### -disable-output \
; RUN:     | FileCheck %s --check-prefixes ALL,ANY
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-lld=ld.lld --tapir-hip-xnack=on \
; RUN:           -passes='kit-cgfb' -cgfb-### -disable-output \
; RUN:     | FileCheck %s --check-prefixes ALL,ON
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-lld=ld.lld --tapir-hip-xnack=off \
; RUN:           -passes='kit-cgfb' -cgfb-### -disable-output \
; RUN:     | FileCheck %s --check-prefixes ALL,OFF
;
; ALL: --plugin-opt=-mcpu={{[^:]+}}
; ANY-NOT: :xnack+
; ANY-NOT: :xnack-
; ON-SAME: :xnack+
; OFF-SAME: :xnack-
