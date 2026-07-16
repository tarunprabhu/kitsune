; Check that opt's command line options make it to the tapir target options.
;
; RUN: opt --tapir=cuda %s -disable-output \
; RUN:     --tapir-gpu-prefetch=false \
; RUN:     --tapir-cuda-arch=sm_72 \
; RUN:     --tapir-cuda-virt-arch=compute_72 \
; RUN:     --tapir-cuda-features="+ptx72" \
; RUN:     --tapir-cuda-runtime-bc="%S/input/nvptx.bc" \
; RUN:     -passes="loop-spawning" -dump-tapir-target-options 2>&1 \
; RUN:     | FileCheck %s -check-prefixes ALL,CHECK
;
; RUN: opt --tapir=cuda %s -disable-output \
; RUN:     --tapir-gpu-prefetch=true \
; RUN:     -passes="loop-spawning" -dump-tapir-target-options \
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
