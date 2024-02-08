; Check that the code model in the target machine is set correctly when
; generating the fat binary.
;
; At this time, we always use the small code model. It is unlikely that we will
; ever use anything else, or make this configurable.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-lld=ld.lld -disable-output \
; RUN:           -passes='kit-cgfb' -cgfb-debug-target-machine 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: Code model: small
