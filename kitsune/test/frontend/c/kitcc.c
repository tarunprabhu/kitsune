// Check that the Kitsune C frontend does not link any kitsune runtime libraries
// if a -ftapir flag is not been specified. Kokkos mode is not relevant for C,
// so Kokkos libraries should not be linked either, and neither should any C++
// standard libraries.
//
// A default tapir target should *not* be added.

// This test is here because, in the past, we would link the C++ and Kokkos
// libraries indiscriminately, but we don't any longer.

// RUN: %kitcc -### %s 2>&1 | FileCheck %s
// CHECK-NOT: -ftapir
// CHECK-NOT: -fkokkos
// CHECK: "-x" "c"
// CHECK-NOT: -lkit{{.*}}
// CHECK-NOT: "-l{{.*}}c++"
