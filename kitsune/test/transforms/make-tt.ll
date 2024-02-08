; If a tapir target that has not been enabled is requested, ensure that the
; process fails gracefully with an error message. This test is primarily for the
; code in kitsune/lib/Targets/TapirTargets.cpp that cannot be unit-tested. We
; use the loop-spawning pass because that guarantees that Kitsune will attempt
; to construct the tapir target objects.
;
; ------------------------------------------------------------------------------
; RUN: %if kitsune-no-cuda %{ \
; RUN:    not opt --tapir=cuda -passes='loop-spawning' %s \
; RUN:        -disable-output 2>&1 \
; RUN:        | FileCheck %s --check-prefix=CUDA \
; RUN: %}
;
; CUDA: error: required tapir target 'cuda' has not been enabled
;
; ------------------------------------------------------------------------------
; RUN: %if kitsune-no-hip %{ \
; RUN:    not opt --tapir=hip -passes='loop-spawning' %s \
; RUN:        -disable-output 2>&1 \
; RUN:        | FileCheck %s --check-prefix=HIP \
; RUN: %}
;
; HIP: error: required tapir target 'hip' has not been enabled
;
; ------------------------------------------------------------------------------
; RUN: %if kitsune-no-opencilk %{ \
; RUN:    not opt --tapir=opencilk -passes='loop-spawning' %s \
; RUN:        -disable-output 2>&1 \
; RUN:        | FileCheck %s --check-prefix=OPENCILK \
; RUN: %}
;
; OPENCILK: error: required tapir target 'opencilk' has not been enabled
;
; ------------------------------------------------------------------------------
; RUN: %if kitsune-no-qthreads %{ \
; RUN:    not opt --tapir=qthreads -passes='loop-spawning' %s \
; RUN:        -disable-output 2>&1 \
; RUN:        | FileCheck %s --check-prefix=QTHREADS \
; RUN: %}
;
; QTHREADS: error: required tapir target 'qthreads' has not been enabled
;
; ------------------------------------------------------------------------------
; RUN: %if kitsune-no-realm %{ \
; RUN:    not opt --tapir=realm -passes='loop-spawning' %s \
; RUN:        -disable-output 2>&1 \
; RUN:        | FileCheck %s --check-prefix=REALM \
; RUN: %}
;
; REALM: error: required tapir target 'realm' has not been enabled
