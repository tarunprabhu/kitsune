// Kitsune does not support Objective C

// RUN: %clang -c -o /dev/null %s
// RUN: %kitxx -c -o /dev/null %s
// RUN: not %kitxx -ftapir=serial -O1 %s 2>&1 | FileCheck %s

// CHECK: kitsune does not support Objective-C
