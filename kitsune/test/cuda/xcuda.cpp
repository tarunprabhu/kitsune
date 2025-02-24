// -x cuda is not supported in combination with a Tapir target. Check that we
// don't disallow other uses.

// RUN: %clang -### -c -emit-llvm -x cuda -nocudalib -nocudainc %s
// RUN: %kitxx -### -c -emit-llvm -x cuda -nocudalib -nocudainc %s
// RUN: not %kitxx -c -emit-llvm -x cuda -ftapir=cuda -nocudalib -nocudainc %s \
// RUN:     2>&1 | FileCheck %s

// CHECK: kitsune does not support the Cuda language
