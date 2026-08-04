// REQUIRES: nvidia-gpu
//
// Check that the cuda tapir target runtime initializes (and finalizes) the
// other runtime components correctly.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s --match-full-lines
//
// CHECK: kitrt: Initializing Kitsune runtime (common)
// CHECK: kitrt: Initialized Kitsune runtime (common)
// CHECK: kitrt: [cuda]: Initializing Kitsune runtime (cuda)
// CHECK: kitrt: [cuda]: Initialized Kitsune runtime (cuda)
// CHECK: Clarke
// CHECK: kitrt: [cuda]: Finalizing Kitsune runtime (cuda)
// CHECK: kitrt: [cuda]: Finalized Kitsune runtime (cuda)
// CHECK: kitrt: Finalizing Kitsune runtime (common)
// CHECK: kitrt: Finalized Kitsune runtime (common)

#include <stdio.h>

// cuda/kitcuda.h pulls in the hip headers. Since we don't need any of that,
// just declare what we need.
extern "C" void __kitcuda_initialize(void);
extern "C" void __kitcuda_finalize(void);

__attribute__((constructor)) static void ctor(void) { __kitcuda_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitcuda_finalize(); }

int main(int argc, char *argv[]) {
  fprintf(stderr, "Clarke\n");
  return 0;
}
