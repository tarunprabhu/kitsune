; Check that both the --tapir and --tapir-target are valid options for opt.
;
; RUN: opt --tapir=openmp -passes=kit-print-tt-options -disable-output \
; RUN:     | FileCheck %s
;
; RUN: opt --tapir-target=openmp -passes=kit-print-tt-options -disable-output \
; RUN:     | FileCheck %s
;
; CHECK: Primary: openmp
