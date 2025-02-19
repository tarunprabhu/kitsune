// Kitsune does not support .hip files

// RUN: %clang -c -o /dev/null %s
// RUN: %kitxx -c -o /dev/null %s
// RUN: not %kitxx -ftapir=cuda %s 2>&1 | FileCheck %s

// CHECK: kitsune does not support the Hip language
