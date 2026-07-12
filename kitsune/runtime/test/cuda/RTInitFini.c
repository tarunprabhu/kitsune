// REQUIRES: nvidia-gpu
//
// Check that the cuda tapir target runtime initializes (and finalizes) the
// other runtime components correctly.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initializing Kitsune runtime (cuda)
// CHECK: Initializing Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (cuda)
// CHECK: Clarke
// CHECK: Finalizing Kitsune runtime (cuda)
// CHECK: Finalizing Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (cuda)

#include <stdio.h>

// cuda/kitcuda.h is not safe to be included in C source files. That header
// should be modified, but that might be a non-trivial change, so for now, this
// is easier.
void __kitcuda_initialize(void);
void __kitcuda_finalize(void);

__attribute__((constructor)) static void ctor(void) { __kitcuda_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitcuda_finalize(); }

int main(int argc, char *argv[]) {
  fprintf(stderr, "Clarke\n");
  return 0;
}
