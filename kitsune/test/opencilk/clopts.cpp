// Check that the frontend options make it to the tapir target
//
// RUN: %kitxx --tapir=opencilk --tapir-verbose \
// RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,COMPILE
//
// RUN: %kitxx --tapir=opencilk --tapir-verbose --kitrt-verbose \
// RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,RUNTIME
//
// ALL: 'opencilk' tapir target options
// COMPILE:   Runtime verbose: 1
// RUNTIME:   Runtime verbose: 1
// ALL:       Bitcode file: {{.+}}/libopencilk-abi.bc

#include <kitsune.h>

// We need a forall loop so the OpencilkABI is entered.
void f(int *c, int n) {
  forall(int i = 0; i < n; ++i) { c[i] = n; }
}
