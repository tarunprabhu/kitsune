; Check that unknown --emb-O<N> options are handled correctly
;
; RUN: not opt --tapir=cuda --tapir-cuda-runtime-bc=%S/input/libdevice.ll %s \
; RUN:     -passes='emb-optimize' -emb-O4 2>&1 \
; RUN:     | FileCheck %s --check-prefix=O4
;
; O4: Unknown command line argument '-emb-O4'
