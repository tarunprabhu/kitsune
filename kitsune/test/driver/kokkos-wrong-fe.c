// -fkokkos cannot be used with Kitsune's C driver.
//
// RUN: not %kitcc -### --kokkos %s 2>&1 | FileCheck %s
// RUN: not %kitcc -### --kokkos-no-init %s 2>&1 | FileCheck %s
//
// CHECK: option '--kokkos{{.*}}' can only be used with kit++
