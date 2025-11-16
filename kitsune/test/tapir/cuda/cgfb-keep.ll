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
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt --tapir=cuda --tapir-cuda-arch=sm_80  \
; RUN:           --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:           -passes='kit-cgfb' \
; RUN:           -o /dev/null
; RUN: not ls -l %t/kitcu-*-.*
;
; ------------------------------------------------------------------------------
;
; Check that when the -cgfb-keep-files option is used, the temporary files used
; by the various cuda code generation utilities (ptxas, fatbinary etc.) are not
; deleted after use.
;
; RUN: %kit-enc --tapir=cuda %s \
; RUN:     | opt --tapir=cuda --tapir-cuda-arch=sm_80 \
; RUN:           --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:           -passes='kit-cgfb' -cgfb-keep-files \
; RUN:           -o /dev/null
; RUN: ls -l %t/kitcu-*.* | FileCheck %s -check-prefix=EXT
; RUN: ls -l %t/kitcu-*.* | FileCheck %s -check-prefix=COUNT
;
; EXT-DAG: {{[.]cufatbin$}}
; EXT-DAG: {{[.]ptx$}}
; EXT-DAG: {{[.]s$}}
; COUNT-COUNT-3: {{.+}}
;
; ------------------------------------------------------------------------------
