// Check that the timing context is correctly initialized and finalized by the
// runtime.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s --match-full-lines
//
// CHECK: kitrt: Initializing Kitsune runtime (common)
// CHECK: kitrt: Initializing Kitsune timing context
// CHECK: kitrt: Initialized Kitsune timing context
// CHECK: kitrt: Initialized Kitsune runtime (common)
// CHECK: Gaiman
// CHECK: kitrt: Finalizing Kitsune runtime (common)
// CHECK: kitrt: Finalizing Kitsune timing context
// CHECK: kitrt: Finalized Kitsune timing context
// CHECK: kitrt: Finalized Kitsune runtime (common)

#include "common/timer.h"
#include "kitrt.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitrt_finalize(); }

int main(int argc, char *argv[]) {
  fprintf(stderr, "Gaiman\n");
  return 0;
}
