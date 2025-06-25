; Check that unknown --emb-O<N> options are handled correctly
;
; RUN: not opt --tapir=cuda --tapir-cuda-runtime-bc=%S/input/libdevice.ll %s \
; RUN:     -passes='tapir-lowering<O1>,emb-optimize' -o /dev/null \
; RUN:     -emb-print-pipeline-passes -emb-O4 2>&1 \
; RUN:     | FileCheck %s --check-prefix=O4
;
; O4: Unknown command line argument '-emb-O4'
