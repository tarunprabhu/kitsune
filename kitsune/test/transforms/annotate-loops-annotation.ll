; Check that the kit-annotate-prelower pass adds the correct annotation after
; it is run, even if it does nothing.
;
; RUN: opt -passes='kit-annotate-prelower' -S %s \
; RUN:     | FileCheck %s
;
; CHECK: !kit.module.pre.lower.annotate.pass = !{}
