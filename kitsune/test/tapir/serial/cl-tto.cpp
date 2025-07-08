// Check that the frontend options common to all tapir targets make it to the
// tapir target optoins. This is a suitable place to check for these since the
// serial tapir target is guaranteed to be built.
//
// RUN: %kitxx --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     -mllvm -dump-tapir-target-options \
// RUN:     | FileCheck %s -check-prefixes ALL,DEFAULT
//
// RUN: %kitxx --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     -mllvm -dump-tapir-target-options \
// RUN:     --tapir-verbose \
// RUN:     | FileCheck %s -check-prefixes ALL,COMPILER
//
// RUN: %kitxx --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     -mllvm -dump-tapir-target-options \
// RUN:     --kitrt-verbose \
// RUN:     | FileCheck %s -check-prefixes ALL,RUNTIME
//
// RUN: %kitxx --tapir=serial -O1 -S -emit-llvm -o /dev/null %s \
// RUN:     -mllvm -dump-tapir-target-options \
// RUN:     | FileCheck %s --check-prefixes ALL,O1
//
// RUN: %kitxx --tapir=serial -O3 -S -emit-llvm -o /dev/null %s \
// RUN:     -mllvm -dump-tapir-target-options \
// RUN:     | FileCheck %s --check-prefixes ALL,O3
//
// RUN: %kitxx --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     -mllvm -dump-tapir-target-options \
// RUN:     -ffp-contract=off \
// RUN:     | FileCheck %s -check-prefixes ALL,FP_STANDARD
//
// RUN: %kitxx --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     -mllvm -dump-tapir-target-options \
// RUN:     -ffp-contract=on \
// RUN:     | FileCheck %s -check-prefixes ALL,FP_STANDARD
//
// RUN: %kitxx --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     -mllvm -dump-tapir-target-options \
// RUN:     -ffp-contract=fast \
// RUN:     | FileCheck %s -check-prefixes ALL,FP_FAST
//
// RUN: %kitxx --tapir=serial -O2 -S -emit-llvm -o /dev/null %s \
// RUN:     -mllvm -dump-tapir-target-options \
// RUN:     -ffp-contract=fast-honor-pragmas \
// RUN:     | FileCheck %s -check-prefixes ALL,FP_STANDARD
//
// ALL:          Tapir target options
// ALL:          Primary: serial
// DEFAULT:      Compiler verbose: 0
// COMPILER:     Compiler verbose: 1
// COMPILER:     Runtime verbose: 1
// RUNTIME:      Compiler verbose: 0
// RUNTIME:      Runtime verbose: 1
// O1:           Optimization level: O1
// O3:           Optimization level: O3
// FP_STANDARD:  FP fusion: standard
// FP_FAST:      FP fusion: fast
