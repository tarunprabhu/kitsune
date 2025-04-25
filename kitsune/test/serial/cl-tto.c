// Check that the frontend options common to all tapir targets make it to the
// tapir target. This is suitable place to check for these since the serial
// tapir target is guaranteed to be built.
//
// RUN: %kitcc --tapir=serial --tapir-verbose       \
// RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,COMPILE
//
// RUN: %kitcc --tapir=serial --tapir-verbose --kitrt-verbose \
// RUN:     -O2 -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,RUNTIME
//
// RUN: %kitcc --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-verbose -ffp-contract=off 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,FP_STANDARD
//
// RUN: %kitcc --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-verbose -ffp-contract=on 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,FP_STANDARD
//
// RUN: %kitcc --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-verbose -ffp-contract=fast 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,FP_FAST
//
// RUN: %kitcc --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     --tapir-verbose -ffp-contract=fast-honor-pragmas 2>&1 \
// RUN:     | FileCheck %s -check-prefixes ALL,FP_STANDARD
//
// ALL: 'serial' tapir target options
// COMPILE:      Runtime verbose: 1
// RUNTIME:      Runtime verbose: 1
// FP_STANDARD:  FP fusion: standard
// FP_FAST:      FP fusion: fast

// We just need some function to ensure that a tapir target object is created.
void f() {}
