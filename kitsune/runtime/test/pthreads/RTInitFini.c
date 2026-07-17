// Check that the pthreads tapir target runtime initializes (and finalizes) the
// other runtime components correctly.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s --match-full-lines
//
// CHECK: kitrt: [pthreads]: Initializing Kitsune runtime (pthreads)
// CHECK: kitrt: Initializing Kitsune runtime (common)
// CHECK: kitrt: Initialized Kitsune runtime (common)
// CHECK: kitrt: [pthreads]: Initialized Kitsune runtime (pthreads)
// CHECK: Niven
// CHECK: kitrt: [pthreads]: Finalizing Kitsune runtime (pthreads)
// CHECK: kitrt: Finalizing Kitsune runtime (common)
// CHECK: kitrt: Finalized Kitsune runtime (common)
// CHECK: kitrt: [pthreads]: Finalized Kitsune runtime (pthreads)

#include <pthreads/kitpthr.h>

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitpthr_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitpthr_finalize(); }

int main(int argc, char *argv[]) {
  fprintf(stderr, "Niven\n");
  return 0;
}
