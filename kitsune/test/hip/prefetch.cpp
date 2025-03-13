// Check that prefetch function calls are inserted, or not as appropriate
//
// RUN: %kitxx --tapir=hip -S -emit-llvm -O2 -o - %s \
// RUN:     | FileCheck %s -check-prefix PREFETCH
//
// PREFETCH: define {{.+}} @f
// PREFETCH: call {{.+}} @__kithip_mem_gpu_prefetch
// PREFETCH: call {{.+}} @__kithip_launch_kernel
// PREFETCH: ret void
// PREFETCH-NEXT: }
//
// -----------------------------------------------------------------------------
//
// RUN: %kitxx --tapir=hip -S -emit-llvm -O2 -o - %s \
// RUN:     -mllvm -hipabi-prefetch=false \
// RUN:     | FileCheck %s -check-prefix NO-PREFETCH
//
// NO-PREFETCH: define {{.+}} @f
// NO-PREFETCH-NOT: call {{.+}} @__kithip_mem_gpu_prefetch
// NO-PREFETCH: call {{.+}} @__kithip_launch_kernel
// NO-PREFETCH: ret void
// NO-PREFETCH-NEXT: }

#include <kitsune.h>

extern "C" void f(int *c, size_t n) {
  forall(size_t i = 0; i < n; ++i) c[i] = n;
}
