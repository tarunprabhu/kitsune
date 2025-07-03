// --tapir= must be provided if either --kokkos or --kokkos-no-init are used.
//
// RUN: not %kitxx -### --kokkos 2>&1 | FileCheck %s
// RUN: not %kitxx -### --kokkos-no-init 2>&1 | FileCheck %s
//
// CHECK: error: --tapir is required with '--kokkos{{.*}}'
