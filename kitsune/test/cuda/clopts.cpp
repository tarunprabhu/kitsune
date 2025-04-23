// Check that the options provided to the frontend make it to the tapir target.
// This is restricted to the options available in the main driver. cc1 requires
// some options but those are not tested here.
//
// On some systems, auto-detecting an NVIDIA GPU takes over 1 second which can
// really add up. So just provide an architecture to have these run fast.
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose --tapir-cuda-arch=sm_60 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,ARCH
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose --tapir-threads-per-block=64 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,TPB
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose --tapir-max-threads-per-block=64 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,MTPB
//
// ALL: 'cuda' tapir target options
// ARCH:  GPU arch: sm_60
// TPB:   Fixed threads/block: 64
// MTPB:  Max threads/block: 64

// We just need some function to ensure that a tapir target object is created.
void f() {}
