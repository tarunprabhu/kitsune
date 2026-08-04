// Check that the openmp tapir target runtime initializes (and finalizes) the
// other runtime components correctly.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s --match-full-lines
//
// CHECK: kitrt: Initializing Kitsune runtime (common)
// CHECK: kitrt: Initialized Kitsune runtime (common)
// CHECK: kitrt: [openmp]: Initializing Kitsune runtime (openmp)
// CHECK: kitrt: [openmp]: Initialized Kitsune runtime (openmp)
// CHECK: Herbert
// CHECK: kitrt: [openmp]: Finalizing Kitsune runtime (openmp)
// CHECK: kitrt: [openmp]: Finalized Kitsune runtime (openmp)
// CHECK: kitrt: Finalizing Kitsune runtime (common)
// CHECK: kitrt: Finalized Kitsune runtime (common)

#include <openmp/kitomp.h>

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitomp_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitomp_finalize(); }

int main(int argc, char *argv[]) {
  fprintf(stderr, "Herbert\n");
  return 0;
}
