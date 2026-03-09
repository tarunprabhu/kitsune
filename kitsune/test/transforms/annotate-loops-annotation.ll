; Check that the annotate-tapir-loops pass adds the correct annotation after it
; is run, even if it does nothing.
;
; RUN: opt -passes='kit-annotate-tapir-loops' -S %s \
; RUN:     | FileCheck %s
;
; CHECK: !kit.module.loops.annotated = !{}
