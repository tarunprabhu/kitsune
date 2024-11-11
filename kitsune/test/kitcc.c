// Check that the Kitsune C frontend behaves like a C frontend and does not
// link a C++ library, nor does it link any kitsune-related libraries.
//
// For now, this should not add any -ftapir target but that might change in the
// future.

// RUN: %kitcc -### %s 2>&1 | FileCheck %s
// CHECK-NOT: -ftapir
// CHECK-NOT: -fkokkos
// CHECK: "-x" "c"
// CHECK-NOT: -lkitrt
// CHECK-NOT: "-l{{.*}}c++"
