; Check that the tapir target options specific to the cuda tapir target are set
; correctly.
;
; RUN: opt --tapir=cuda %s -disable-output \
; RUN:     --tapir-gpu-prefetch=false \
; RUN:     --tapir-cuda-arch=sm_72 \
; RUN:     --tapir-cuda-virt-arch=compute_72 \
; RUN:     --tapir-cuda-features="+ptx72" \
; RUN:     --tapir-cuda-runtime-bc="%S/input/nvptx.bc" \
; RUN:     -passes="kit-print-tt-options" 2>&1 \
; RUN:     | FileCheck %s -check-prefixes ALL,CHECK
;
; RUN: opt --tapir=cuda %s -disable-output \
; RUN:     --tapir-gpu-prefetch=true \
; RUN:     -passes="kit-print-tt-options" \
; RUN:     | FileCheck %s -check-prefixes ALL,PREFETCH
;
; ALL:       Tapir target options
; ALL:       Primary: cuda
; CHECK:     GPU prefetch: 0
; CHECK:     Cuda arch: sm_72
; CHECK:     Cuda virtual arch: compute_72
; CHECK:     Cuda target features: +ptx72
; CHECK:     Cuda bitcode file: {{.+}}/input/nvptx.bc
; PREFETCH:  GPU prefetch: 1
