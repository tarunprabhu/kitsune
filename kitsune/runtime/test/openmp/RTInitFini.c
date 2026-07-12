// Check that the openmp tapir target runtime initializes (and finalizes) the
// other runtime components correctly.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initializing Kitsune runtime (openmp)
// CHECK: Initializing Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (openmp)
// CHECK: Herbert
// CHECK: Finalizing Kitsune runtime (openmp)
// CHECK: Finalizing Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (openmp)

#include <openmp/kitomp.h>

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitomp_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitomp_finalize(); }

int main(int argc, char *argv[]) {
  fprintf(stderr, "Herbert\n");
  return 0;
}
