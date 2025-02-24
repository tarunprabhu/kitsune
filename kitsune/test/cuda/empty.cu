// Kitsune does not support .cu files

// RUN: %clang -c -emit-llvm -nocudalib -nocudainc %s
// RUN: %kitxx -c -emit-llvm -nocudalib -nocudainc %s
// RUN: not %kitxx -c -emit-llvm -ftapir=cuda -nocudalib -nocudainc %s 2>&1 \
// RUN:     | FileCheck %s

// CHECK: kitsune does not support the Cuda language
