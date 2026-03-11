; Check that opt's command line options make it to the tapir target options.
;
; RUN: opt --tapir=cuda -passes="loop-spawning" -o /dev/null %s \
; RUN:     -dump-tapir-target-options \
; RUN:     --tapir-gpu-tpb=64 \
; RUN:     --tapir-gpu-max-tpb=128 \
; RUN:     --tapir-gpu-prefetch=false \
; RUN:     --tapir-cuda-arch=sm_72 \
; RUN:     --tapir-cuda-virt-arch=compute_72 \
; RUN:     --tapir-cuda-features="+ptx72" \
; RUN:     --tapir-cuda-runtime-bc="%S/input/nvptx.bc" 2>&1\
; RUN:     | FileCheck %s -check-prefixes ALL,CHECK
;
; RUN: opt --tapir=cuda --tapir-cuda-arch=sm_86 \
; RUN:     --tapir-cuda-runtime-bc=%S/input/libdevice.ll \
; RUN:     --tapir-gpu-prefetch=true 2>&1 \
; RUN:     -passes="loop-spawning" -o /dev/null %s \
; RUN:     -dump-tapir-target-options \
; RUN:     | FileCheck %s -check-prefixes ALL,PREFETCH
;
; ALL:       Tapir target options
; ALL:       Primary: cuda
; CHECK:     GPU fixed threads/block: 64
; CHECK:     GPU max threads/block: 128
; CHECK:     GPU prefetch: 0
; CHECK:     Cuda arch: sm_72
; CHECK:     Cuda virtual arch: compute_72
; CHECK:     Cuda target features: +ptx72
; CHECK:     Cuda bitcode file: {{.+}}/input/nvptx.bc
; PREFETCH:  GPU prefetch: 1
