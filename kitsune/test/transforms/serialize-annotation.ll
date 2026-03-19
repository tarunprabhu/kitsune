; Check that the kit-serialize pass adds the correct annotation after it is run,
; even if it does nothing.
;
; RUN: opt -passes='kit-serialize' -S %s \
; RUN:     | FileCheck %s
;
; CHECK: !kit.module.serialize.pass = !{}

!kit.module.pre.lower.annotate.pass = !{}
