// -x hip is not supported in combination with a Tapir target. Check that we
// don't disallow other uses.

// RUN: %clang -c -x hip %s
// RUN: %kitxx -c -x hip %s
// RUN: not %kitxx -x hip -ftapir=hip %s 2>&1 | FileCheck %s

// CHECK: kitsune does not support the Hip language
