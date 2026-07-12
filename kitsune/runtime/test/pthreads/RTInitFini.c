// Check that the pthreads tapir target runtime initializes (and finalizes) the
// other runtime components correctly.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initializing Kitsune runtime (pthreads)
// CHECK: Initializing Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (pthreads)
// CHECK: Niven
// CHECK: Finalizing Kitsune runtime (pthreads)
// CHECK: Finalizing Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (pthreads)

#include <pthreads/kitpthr.h>

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitpthr_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitpthr_finalize(); }

int main(int argc, char *argv[]) {
  fprintf(stderr, "Niven\n");
  return 0;
}
