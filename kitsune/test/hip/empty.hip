// Kitsune does not support .hip files

// RUN: %clang -c -o /dev/null -nogpuinc -nogpulib %s
// RUN: %kitxx -c -o /dev/null -nogpuinc -nogpulib %s
// RUN: not %kitxx -ftapir=cuda -nogpuinc -nogpulib %s 2>&1 | FileCheck %s

// CHECK: kitsune does not support the Hip language
