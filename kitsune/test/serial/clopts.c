// RUN: %kitcc --tapir=serial --tapir-verbose \
// RUN:      -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:      | FileCheck %s -check-prefixes ALL,COMPILE
//
// RUN: %kitcc --tapir=serial --tapir-verbose --kitrt-verbose \
// RUN:      -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:      | FileCheck %s -check-prefixes ALL,RUNTIME
//
// ALL: 'serial' tapir target options
// COMPILE:   Runtime verbose:{{[ ]*}} false
// RUNTIME:   Runtime verbose:{{[ ]*}} true

#include <kitsune.h>

// We need a forall loop so the SerialABI is entered.
void f(int *c, int n) {
  forall(int i = 0; i < n; ++i) { c[i] = n; }
}
