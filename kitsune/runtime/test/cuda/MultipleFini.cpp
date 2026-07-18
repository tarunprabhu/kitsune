// REQUIRES: nvidia-gpu
//
// Check that finalizing the runtime multiple times has no effect. The runtime
// should remain finalized.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initialized = {{[1-9][0-9]*}}
// CHECK: Finalizing Kitsune runtime (cuda)
// CHECK: Finalized Kitsune runtime (cuda)
// CHECK: Finalizing Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (common)
// CHECK: Initialized = 0
// CHECK: Cannot finalize runtime. Not initialized
// CHECK: Initialized = 0

#include "cuda/kitcuda.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitcuda_initialize(); }

__attribute__((destructor)) static void dtor(void) {
  fprintf(stderr, "Initialized = %d\n", __kitcuda_is_initialized());
  __kitcuda_finalize();
  fprintf(stderr, "Initialized = %d\n", __kitcuda_is_initialized());
  __kitcuda_finalize();
  fprintf(stderr, "Initialized = %d\n", __kitcuda_is_initialized());
}

int main(int argc, char *argv[]) { return 0; }
