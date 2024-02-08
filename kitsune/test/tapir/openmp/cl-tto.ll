; Check that the opt command line options make it to the tapir target options.
; This is intended for options that are specific to the openmp tapir target.
;
; NOTE: Currently, there are no such options, so this is mostly just a
; placeholder and is around for consistency with the tests for the other tapir
; targets.
;
; RUN: opt --tapir=openmp %s -disable-output \
; RUN:     -passes="loop-spawning" -dump-tapir-target-options \
; RUN:     | FileCheck %s -check-prefixes ALL
;
; ALL:          Tapir target options
; ALL:          Primary: openmp
