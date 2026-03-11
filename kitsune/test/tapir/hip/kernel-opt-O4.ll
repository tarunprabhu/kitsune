; Check that unknown --emb-O<N> options are handled correctly
;
; RUN: not opt --tapir=hip --tapir-hip-runtime-bcs=%S/input/libdevice.ll %s \
; RUN:     -passes='emb-optimize' -emb-O4 2>&1 \
; RUN:     | FileCheck %s --check-prefix=O4
;
; O4: Unknown command line argument '-emb-O4'
