// Check that the qthreads tapir target runtime initializes (and finalizes) the
// other runtime components correctly.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initializing Kitsune runtime (qthreads)
// CHECK: Initializing Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (qthreads)
// CHECK: Asimov
// CHECK: Finalizing Kitsune runtime (qthreads)
// CHECK: Finalizing Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (qthreads)

#include <qthreads/kitqthr.h>

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitqthr_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitqthr_finalize(); }

int main(int argc, char *argv[]) {
  fprintf(stderr, "Asimov\n");
  return 0;
}
