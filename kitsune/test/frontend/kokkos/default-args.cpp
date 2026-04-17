// RUN: %kitxx -### -fkokkos --tapir=nolo %s 2>&1 | FileCheck %s
// RUN: %kitxx -### --kokkos --tapir=nolo %s 2>&1 | FileCheck %s
// RUN: %kitxx -### -fkokkos-no-init --tapir=nolo %s 2>&1 | FileCheck %s
// RUN: %kitxx -### --kokkos-no-init --tapir=nolo %s 2>&1 | FileCheck %s
//
// CHECK: "-cc1"
// CHECK-SAME: -I{{[^ ]*}}/include/kokkos
//
// The next line is expected to be the linker invocation. Since it is difficult
// to reliably check the name of the linker executable, just check for the
// expected linker flags.
//
// CHECK: -lkokkoscore
// CHECK-SAME: -lkitrt
