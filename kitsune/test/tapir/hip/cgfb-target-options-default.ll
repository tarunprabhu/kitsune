; When creating the target machine to be used for code generation, some target
; options are set explicitly. Check that these options are as expected.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-lld=ld.lld \
; RUN:           -passes='kit-cgfb' -disable-output \
; RUN:           -cgfb-debug-target-options 2>&1 \
; RUN:     | FileCheck %s
;
; CHECK: AllowFPOpFusion: standard
; CHECK: EmitAddrsig: true
; CHECK: UseInitArray: true
