// Check that the frontend options common to all tapir targets make it to the
// tapir target. This is suitable place to check for these since the serial
// tapir target is guaranteed to be built.
//
// RUN: %kitxx --tapir=serial --tapir-verbose \
// RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,COMPILE
//
// RUN: %kitxx --tapir=serial --tapir-verbose --kitrt-verbose \
// RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,RUNTIME
//
// RUN: %kitxx --tapir=serial --tapir-verbose -O1 -S -emit-llvm -o - %s 2>&1 \
// RUN:     | FileCheck %s --check-prefixes ALL,O1
//
// RUN: %kitxx --tapir=serial --tapir-verbose -O3 -S -emit-llvm -o - %s 2>&1 \
// RUN:     | FileCheck %s --check-prefixes ALL,O3
//
// RUN: %kitxx --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-verbose -ffp-contract=off 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,FP_STANDARD
//
// RUN: %kitxx --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-verbose -ffp-contract=on 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,FP_STANDARD
//
// RUN: %kitxx --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-verbose -ffp-contract=fast 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,FP_FAST
//
// RUN: %kitxx --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-verbose -ffp-contract=fast-honor-pragmas 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,FP_STANDARD
//
// ALL: 'serial' tapir target options
// COMPILE:      Runtime verbose: 1
// RUNTIME:      Runtime verbose: 1
// O1:           Optimization level: O1
// O3:           Optimization level: O3
// FP_STANDARD:  FP fusion: standard
// FP_FAST:      FP fusion: fast

// We just need some function to ensure that a tapir target object is created.
void f() {}
