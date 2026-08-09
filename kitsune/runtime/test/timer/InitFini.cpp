// Check that the timing context is correctly initialized and finalized by the
// runtime.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s --match-full-lines
//
// CHECK: kitrt: Initializing Kitsune runtime (common)
// CHECK: kitrt: Initialized Kitsune runtime (common)
// CHECK: kitrt: Initializing Kitsune runtime (timer)
// CHECK: kitrt: Initialized Kitsune runtime (timer)
// CHECK: Gaiman
// CHECK: kitrt: Finalizing Kitsune runtime (timer)
// CHECK: kitrt: Finalized Kitsune runtime (timer)
// CHECK: kitrt: Finalizing Kitsune runtime (common)
// CHECK: kitrt: Finalized Kitsune runtime (common)

#include "TestHelpers.h"

#include <stdio.h>

CTOR(RT_TIMER)

int main(int argc, char *argv[]) {
  fprintf(stderr, "Gaiman\n");
  return 0;
}
