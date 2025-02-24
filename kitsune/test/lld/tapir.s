; RUN: not ld.lld --tapir-target=bad 2>&1 \
; RUN:     | FileCheck %s -check-prefix BAD
; RUN: not ld.lld --tapir-target= 2>&1 \
; RUN:     | FileCheck %s -check-prefix MISSING

; BAD: invalid value 'bad' in '--tapir-target'
; MISSING: invalid value '' in '--tapir-target'
