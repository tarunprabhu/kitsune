// -----------------------------------------------------------------------------
// Check that the options provided to kit++ make it to the tapir target.
//
// On some systems, auto-detecting an NVIDIA GPU takes over 1 second which can
// really add up. So just provide an architecture to have these run fast.
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose --tapir-threads-per-block=64 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,TPB
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose --tapir-max-threads-per-block=128 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,MTPB
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose --tapir-gpu-prefetch 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,PREFETCH
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose --tapir-gpu-no-prefetch 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,NO-PREFETCH
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-verbose --tapir-cuda-arch=sm_60 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,ARCH
//
// -----------------------------------------------------------------------------
// Check that the options only allowed in -cc1 make it to the tapir targets.
//
// RUN: %kitxx -cc1 --tapir=cuda --tapir-verbose -O2 %s -o /dev/null \
// RUN:     -disable-free -emit-llvm \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-cuda-virt-arch=compute_72 \
// RUN:     --tapir-cuda-features="+ptx72" \
// RUN:     --tapir-cuda-runtime-bc="%S/input/nvptx.bc" 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,CC1
//
// -----------------------------------------------------------------------------
// ALL: 'cuda' tapir target options
// TPB:           GPU fixed threads/block: 64
// MTPB:          GPU max threads/block: 128
// PREFETCH:      GPU prefetch: 1
// NO-PREFETCH:   GPU prefetch: 0
// ARCH:          Cuda arch: sm_60
// CC1:           Cuda virtual arch: compute_72
// CC1:           Cuda target features: +ptx72
// CC1:           Cuda bitcode file: {{.+}}.bc

// We just need some function to ensure that a tapir target object is created.
void f() {}
