; Check that the intermediate files generated during codegen are not deleted if
; the -cgfb-keep-files option is provided.
;
; ------------------------------------------------------------------------------
; RUN: rm -rf %t
; RUN: mkdir -p %t
; RUN: export TMPDIR=%t
; RUN: export TEMP=%t
; RUN: export TMP=%t
;
; ------------------------------------------------------------------------------
;
; Check that the intermediate files are cleaned up by default.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-lld=ld.lld -disable-output \
; RUN:            -passes='kit-cgfb'
; RUN: not ls -l %t/kithip-*.*
;
; ------------------------------------------------------------------------------
;
; Check that when the -cgfb-keep-files option is used, the intermediate files
; are not deleted after use.
;
; RUN: %kit-enc --tapir=hip %s \
; RUN:     | opt --tapir=hip --tapir-lld=ld.lld -disable-output \
; RUN:           -passes='kit-cgfb' -cgfb-keep-files
; RUN: ls -l %t/kithip-*.* | FileCheck %s -check-prefix=EXT
; RUN: ls -l %t/kithip-*.* | FileCheck %s -check-prefix=COUNT
;
; EXT-DAG: {{[.]amdgpu.o}}
; EXT-DAG: {{[.]amdgpu.so}}
; COUNT-COUNT-2: {{.+}}
;
; ------------------------------------------------------------------------------
