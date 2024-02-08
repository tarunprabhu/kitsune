// Check that the Kitsune C++ frontend does not link any kitsune runtime
// libraries if a -ftapir flag is not been specified. Kokkos mode will never be
// enabled by default, and if it is not, Kokkos libraries should not be linked
// either.
//
// A default tapir target should *not* be added.

// This test is here because, in the past, we would link the Kokkos libraries
// indiscriminately, but we don't any longer.

// RUN: %kitxx -### %s 2>&1 | FileCheck %s
// CHECK-NOT: -ftapir
// CHECK-NOT: -fkokkos
// CHECK: "-x" "c++"
// CHECK-NOT: -lkit{{.+}}
// CHECK: "-l{{.*}}c++"
