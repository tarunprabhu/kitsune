// -x cuda is not supported in combination with a Tapir target. Check that we
// don't disallow other uses.

// RUN: %clang -### -x cuda %s
// RUN: %kitxx -### -x cuda %s
// RUN: not %kitxx -x cuda -ftapir=cuda %s 2>&1 | FileCheck %s

// CHECK: kitsune does not support the Cuda language
