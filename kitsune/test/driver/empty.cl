// Kitsune does not support OpenCL

// RUN: %clang -c -o /dev/null %s
// RUN: %kitxx -c -o /dev/null %s
// RUN: not %kitxx -ftapir=serial %s 2>&1 | FileCheck %s

// CHECK: kitsune does not support OpenCL
