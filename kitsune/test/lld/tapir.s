; RUN: not ld.lld --tapir=bad 2>&1 | FileCheck %s -check-prefix BAD
; RUN: not ld.lld --tapir= 2>&1 | FileCheck %s -check-prefix MISSING

; BAD: invalid value 'bad' in '--tapir'
; MISSING: invalid value '' in '--tapir'
