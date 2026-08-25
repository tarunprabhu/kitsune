; Check that the tapir target options are set correctly depending on the options
; passed to LLVM's opt utility. These options are common to all tapir targets
; Since the serial tapir target is guaranteed to be built, we use that here.
;
; RUN: opt --tapir=serial %s -disable-output \
; RUN:     -passes="tapir-lowering<O2>,kit-print-tt-options" \
; RUN:     | FileCheck %s -check-prefixes ALL
;
; RUN: opt --tapir=serial %s -disable-output \
; RUN:     -passes="tapir-lowering<O1>,kit-print-tt-options" \
; RUN:     | FileCheck %s --check-prefixes ALL,O1
;
; RUN: opt --tapir=serial %s -disable-output \
; RUN:     -passes="tapir-lowering<O3>,kit-print-tt-options" \
; RUN:     | FileCheck %s --check-prefixes ALL,O3
;
; RUN: opt --tapir=serial %s -disable-output \
; RUN:     -fp-contract=off \
; RUN:     -passes="tapir-lowering<O2>,kit-print-tt-options" \
; RUN:     | FileCheck %s -check-prefixes ALL,FP_STRICT
;
; RUN: opt --tapir=serial %s -disable-output \
; RUN:     -fp-contract=on \
; RUN:     -passes="tapir-lowering<O2>,kit-print-tt-options" \
; RUN:     | FileCheck %s -check-prefixes ALL,FP_STANDARD
;
; RUN: opt --tapir=serial %s -disable-output \
; RUN:     -fp-contract=fast \
; RUN:     -passes="tapir-lowering<O2>,kit-print-tt-options" \
; RUN:     | FileCheck %s -check-prefixes ALL,FP_FAST
;
; ALL:          Tapir target options
; O1:           Optimization level: O1
; O3:           Optimization level: O3
; FP_STRICT:    FP fusion: strict
; FP_STANDARD:  FP fusion: standard
; FP_FAST:      FP fusion: fast
;
; ------------------------------------------------------------------------------
; If the --tapir options is not given, we can still ask for the options to be
; printed.
;
; RUN: opt -passes=kit-print-tt-options -disable-output \
; RUN:     | FileCheck %s --check-prefix=NOTAPIR
;
; NOTAPIR: Tapir target options:
; NOTAPIR-NOT: {{^.+$}}
