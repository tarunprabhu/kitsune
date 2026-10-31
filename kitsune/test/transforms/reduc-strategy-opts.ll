; ------------------------------------------------------------------------------
; Check known valid values that are allowed for the kit-cpu-reduce-strategy
; option.
;
; RUN: opt --kit-cpu-reduce-strategy=naive %s -disable-output \
; RUN:     | FileCheck %s --check-prefix=VALID --allow-empty
;
; RUN: opt --kit-cpu-reduce-strategy=direct %s -disable-output \
; RUN:     | FileCheck %s --check-prefix=VALID --allow-empty
;
; RUN: opt --kit-cpu-reduce-strategy=local %s -disable-output \
; RUN:     | FileCheck %s --check-prefix=VALID --allow-empty
;
; ------------------------------------------------------------------------------
; Reduction strategies that are only valid for GPU-centric tapir targets are
; not valid inputs to --kit-cpu-reduce-strategy. "auto" cannot be requested as
; an explicit strategy.
;
; RUN: not opt --kit-cpu-reduce-strategy=auto %s -disable-output 2>&1 \
; RUN:     | FileCheck %s --check-prefix=INVALID-CPU
;
; RUN: not opt --kit-cpu-reduce-strategy=shmem %s -disable-output 2>&1 \
; RUN:     | FileCheck %s --check-prefix=INVALID-CPU
;
; RUN: not opt --kit-cpu-reduce-strategy=wshfl %s -disable-output 2>&1 \
; RUN:     | FileCheck %s --check-prefix=INVALID-CPU
;
; RUN: not opt --kit-cpu-reduce-strategy=wshflshmem %s -disable-output 2>&1 \
; RUN:     | FileCheck %s --check-prefix=INVALID-CPU
;
; ------------------------------------------------------------------------------
; A value must be provided to the --kit-cpu-reduce-strategy option.
;
; RUN: not opt --kit-cpu-reduce-strategy= %s -disable-output 2>&1 \
; RUN:     | FileCheck %s --check-prefix=INVALID-CPU
;
; ------------------------------------------------------------------------------
; Check known valid values that are allowed for the kit-gpu-reduce-strategy
; option.
;
; RUN: opt --kit-gpu-reduce-strategy=naive %s -disable-output \
; RUN:     | FileCheck %s --check-prefix=VALID --allow-empty
;
; RUN: opt --kit-gpu-reduce-strategy=direct %s -disable-output \
; RUN:     | FileCheck %s --check-prefix=VALID --allow-empty
;
; RUN: opt --kit-gpu-reduce-strategy=shmem %s -disable-output \
; RUN:     | FileCheck %s --check-prefix=VALID --allow-empty
;
; RUN: opt --kit-gpu-reduce-strategy=wshfl %s -disable-output \
; RUN:     | FileCheck %s --check-prefix=VALID --allow-empty
;
; RUN: opt --kit-gpu-reduce-strategy=wshflshmem %s -disable-output \
; RUN:     | FileCheck %s --check-prefix=VALID --allow-empty
;
; ------------------------------------------------------------------------------
; Reduction strategies that are only valid for CPU-centric tapir targets are
; not valid inputs to --kit-gpu-reduce-strategy. "auto" cannot be requested as
; an explicit strategy.
;
; RUN: not opt --kit-gpu-reduce-strategy=auto %s -disable-output 2>&1 \
; RUN:     | FileCheck %s --check-prefix=INVALID-GPU
;
; RUN: not opt --kit-gpu-reduce-strategy=local %s -disable-output 2>&1 \
; RUN:     | FileCheck %s --check-prefix=INVALID-GPU
;
; ------------------------------------------------------------------------------
; A value must be provided to the --kit-gpu-reduce-strategy option.
;
; RUN: not opt --kit-gpu-reduce-strategy= %s -disable-output 2>&1 \
; RUN:     | FileCheck %s --check-prefix=INVALID-GPU

; VALID-NOT: {{^.+$}}
; INVALID-CPU: for the --kit-cpu-reduce-strategy option: Cannot find option
; INVALID-GPU: for the --kit-gpu-reduce-strategy option: Cannot find option
