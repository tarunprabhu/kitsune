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
// ALL:       Optimization level: O2
// ALL:       FP Fusion: standard
// ALL:       Bitcode file: {{.+}}/libopencilk-abi.bc

// We just need some function to ensure that a tapir target object is created.
void f() {}
