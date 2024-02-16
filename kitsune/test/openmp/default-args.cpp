// RUN: %kitxx -### -ftapir=openmp %s 2>&1 | FileCheck %s

// The link line may have some optional space at the start of the line followed
// by the absolute path to the linker in quotes. The linker name itself could
// be lld, but we also allow matches to ld.gold, ld.bfd etc.
// CHECK: {{^[ ]*"[^"]+/[l]?}}ld{{[.]?[^ ]*}}"
// CHECK-SAME: -lomp
// CHECK-SAME: -lkitrt
