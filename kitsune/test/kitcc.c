// Check that the Kitsune C frontend behaves like a C frontend and does not
// link a C++ library.
//
// For now, this should not add any -ftapir target but that might change in the
// future. -fkokkos and related flags will never be the default.

// RUN: %kitcc -### %s 2>&1 | FileCheck %s
// CHECK-NOT: -ftapir
// CHECK-NOT: -fkokkos
// CHECK: "-x" "c"
// CHECK-NOT: "-l{{.*}}c++"
