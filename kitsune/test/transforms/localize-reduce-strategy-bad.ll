; Check the allowed values for -kit-cpu-reduce-strategy and
; -kit-gpu-reduce-strategy.
;
; ------------------------------------------------------------------------------
; RUN: not opt -passes='kit-reduce-localize' -kit-cpu-reduce-strategy= \
; RUN:     -disable-output %s 2>&1 \
; RUN:     | FileCheck %s
;
; RUN: not opt -passes='kit-reduce-localize' -kit-cpu-reduce-strategy=auto \
; RUN:     -disable-output %s 2>&1 \
; RUN:     | FileCheck %s
;
; RUN: not opt -passes='kit-reduce-localize' -kit-cpu-reduce-strategy=naive \
; RUN:     -disable-output %s 2>&1 \
; RUN:     | FileCheck %s
;
; ------------------------------------------------------------------------------
; RUN: not opt -passes='kit-reduce-localize' -kit-gpu-reduce-strategy= \
; RUN:     -disable-output %s 2>&1 \
; RUN:     | FileCheck %s
;
; RUN: not opt -passes='kit-reduce-localize' -kit-gpu-reduce-strategy=auto \
; RUN:     -disable-output %s 2>&1 \
; RUN:     | FileCheck %s
;
; RUN: not opt -passes='kit-reduce-localize' -kit-gpu-reduce-strategy=naive \
; RUN:     -disable-output %s 2>&1 \
; RUN:     | FileCheck %s
;
;
; CHECK: Cannot find option named '{{.*}}'
