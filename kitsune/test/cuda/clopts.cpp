// Check that the options provided to the frontend make it to the tapir target.
// This is restricted to the options provided to the driver. cc1 requires some
// options but those are not tested here.
//
// On some systems, auto-detecting an NVIDIA GPU takes over 1 second which can
// really add up. So just provide an architecture to have these run fast.
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,COMPILE
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose -O3 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,OPTLEVEL
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose --kitrt-verbose 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,RUNTIME
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose -ffp-contract=off 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,FP_STANDARD
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose -ffp-contract=on 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,FP_STANDARD
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose -ffp-contract=fast 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,FP_FAST
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose -ffp-contract=fast-honor-pragmas 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,FP_STANDARD
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose --tapir-cuda-arch=sm_60 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,ARCH
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose --tapir-threads-per-block=64 2>&1 \
// RUN:     | FileCheck %s -check-prefix TPB
//
// RUN: %kitxx --tapir=cuda -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-cuda-arch=sm_72 \
// RUN:     --tapir-verbose --tapir-max-threads-per-block=64 2>&1 \
// RUN:     | FileCheck %s -check-prefix MTPB
//
// ALL: 'cuda' tapir target options
// COMPILE:      Runtime verbose: 1
// OPTLEVEL:     Optimization level: O3
// RUNTIME:      Runtime verbose: 1
// ARCH:         GPU arch: sm_60
// TPB:          Fixed threads/block: 64
// MTPB:         Max threads/block: 64
// FP_STANDARD:  FP Fusion: standard
// FP_FAST:      FP Fusion: fast

#include <kitsune.h>

// We need a forall loop so CudaABI is entered.
void f(int *c, int n) {
  forall(int i = 0; i < n; ++i) { c[i] = n; }
}
