; If embedded bitcode is present, a corresponding global containing device code
; must be present as well.
;
; In these tests, kit-enc will generate an embedded bitcode module as well as a
; device code global. The `sed` command will delete the device code global.
;
; ------------------------------------------------------------------------------
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | sed '/kit.emb.fb/d' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck --check-prefix=CUDA %s
;
; CUDA: missing device code global for tapir target 'cuda'
;
; ------------------------------------------------------------------------------
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | sed '/kit.emb.fb/d' \
; RUN:     | not llvm-as -o /dev/null 2>&1 \
; RUN:     | FileCheck --check-prefix=HIP %s
;
; HIP: missing device code global for tapir target 'hip'
;
; ------------------------------------------------------------------------------
