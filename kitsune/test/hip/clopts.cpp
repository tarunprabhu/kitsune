// Check that clang command line options specific to the hip tapir target make
// their way to HipABI.
//
// RUN: %kitxx --tapir=hip --tapir-verbose          \
// RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,COMPILE
//
// RUN: %kitxx --tapir=hip --tapir-verbose --kitrt-verbose \
// RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,RUNTIME
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-hip-arch=gfx906 \
// RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,ARCH
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-threads-per-block=64 %s \
// RUN:     -O2 -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s -check-prefix TPB
//
// RUN: %kitxx --tapir=hip --tapir-verbose --tapir-max-threads-per-block=64 %s \
// RUN:     -O2 -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %s -check-prefix MTPB
//
// ALL: 'hip' tapir target options
// COMPILE:   Runtime verbose: true
// RUNTIME:   Runtime verbose: true
// ARCH:      GPU arch: gfx906
// TPB:       Fixed threads/block: 64
// MTPB:      Max threads/block: 64

#include <kitsune.h>

// We need a forall loop so the HipABI is entered.
void f(int *c, int n) {
  forall(int i = 0; i < n; ++i) { c[i] = n; }
}
