// Check that the qthreads tapir target runtime initializes (and finalizes) the
// other runtime components correctly.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s --match-full-lines
//
// CHECK: kitrt: [qthreads]: Initializing Kitsune runtime (qthreads)
// CHECK: kitrt: Initializing Kitsune runtime (common)
// CHECK: kitrt: Initialized Kitsune runtime (common)
// CHECK: kitrt: [qthreads]: Initialized Kitsune runtime (qthreads)
// CHECK: Asimov
// CHECK: kitrt: [qthreads]: Finalizing Kitsune runtime (qthreads)
// CHECK: kitrt: Finalizing Kitsune runtime (common)
// CHECK: kitrt: Finalized Kitsune runtime (common)
// CHECK: kitrt: [qthreads]: Finalized Kitsune runtime (qthreads)

#include <qthreads/kitqthr.h>

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitqthr_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitqthr_finalize(); }

int main(int argc, char *argv[]) {
  fprintf(stderr, "Asimov\n");
  return 0;
}
