// RUN: %kitxx --tapir=cuda --tapir-verbose \
// RUN:      -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:      | FileCheck %s -check-prefixes ALL,COMPILE
//
// RUN: %kitxx --tapir=cuda --tapir-verbose --kitrt-verbose \
// RUN:      -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:      | FileCheck %s -check-prefixes ALL,RUNTIME
//
// RUN: %kitxx --tapir=cuda --tapir-verbose --tapir-cuda-arch=sm_72 \
// RUN:      -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:      | FileCheck %s -check-prefixes ALL,ARCH
//
// RUN: %kitxx --tapir=cuda --tapir-verbose --tapir-threads-per-block=64 %s \
// RUN:      -O2 -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s -check-prefix TPB
//
// RUN: %kitxx --tapir=cuda --tapir-verbose --tapir-max-threads-per-block=64 %s \
// RUN:      -O2 -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s -check-prefix MTPB
//
// ALL: 'cuda' tapir target options
// COMPILE:   Runtime verbose: false
// RUNTIME:   Runtime verbose: true
// ARCH:      GPU arch: sm_72
// TPB:       Fixed threads/block: 64
// MTPB:      Max threads/block: 64

#include <kitsune.h>

// We need a forall loop so the CudaABI is entered.
void f(int *c, int n) {
  forall(int i = 0; i < n; ++i) { c[i] = n; }
}
