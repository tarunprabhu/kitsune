; A tapir target that does not generate embedded bitcode cannot be used as a
; filter.
;
; RUN: not kit-mbc --tapir nolo %s 2>&1 \
; RUN:     | FileCheck %s -check-prefix NOLO
;
; RUN: not kit-mbc --tapir opencilk %s 2>&1 \
; RUN:     | FileCheck %s -check-prefix OPENCILK
;
; RUN: not kit-mbc --tapir serial %s 2>&1 \
; RUN:     | FileCheck %s -check-prefix SERIAL
;
; NOLO: error: 'nolo' tapir target does not generate embedded bitcode
; SERIAL: error: 'serial' tapir target does not generate embedded bitcode
; OPENCILK: error: 'opencilk' tapir target does not generate embedded bitcode
