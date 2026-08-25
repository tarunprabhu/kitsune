// -----------------------------------------------------------------------------
// Check that the options provided to kit++ make it to the tapir target options.
//
// On some systems, auto-detecting an NVIDIA GPU takes over 1 second which can
// really add up. So just provide an architecture to have these run fast.
//
// RUN: %kitxx --tapir=cuda -O1 -S -emit-llvm -o /dev/null %s \
// RUN:     -mllvm -print-tt-options \
// RUN:     --tapir-cuda-arch=sm_72 --tapir-gpu-prefetch 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,PREFETCH
//
// RUN: %kitxx --tapir=cuda -O1 -S -emit-llvm -o /dev/null %s \
// RUN:     -mllvm -print-tt-options \
// RUN:     --tapir-cuda-arch=sm_72 --tapir-gpu-no-prefetch 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,NO-PREFETCH
//
// RUN: %kitxx --tapir=cuda -O1 -S -emit-llvm -o /dev/null %s \
// RUN:     -mllvm -print-tt-options \
// RUN:     --tapir-cuda-arch=sm_60 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,ARCH
//
// -----------------------------------------------------------------------------
// Check that the options only allowed in -cc1 make it to the tapir targets.
//
// RUN: %kitxx -cc1 --tapir=cuda -O1 %s -o /dev/null \
// RUN:     -mllvm -print-tt-options \
// RUN:     -disable-free -emit-llvm \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-cuda-virt-arch=compute_72 \
// RUN:     --tapir-cuda-features="+ptx72" \
// RUN:     --tapir-cuda-runtime-bc="%S/input/nvptx.bc" 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,CC1
//
// -----------------------------------------------------------------------------
// ALL:          Tapir target options
// ALL:          Primary: cuda
// PREFETCH:     GPU prefetch: 1
// NO-PREFETCH:  GPU prefetch: 0
// ARCH:         Cuda arch: sm_60
// CC1:          Cuda virtual arch: compute_72
// CC1:          Cuda target features: +ptx72
// CC1:          Cuda bitcode file: {{.+}}.bc
