// Check that finalizing the common runtime multiple times has no effect.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Finalizing Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (common)
// CHECK: Cannot finalize runtime. Not initialized
// CHECK-NOT: Finalizing Kitsune runtime

#include "kitrt.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitrt_initialize(); }

__attribute__((destructor)) static void dtor(void) {
  __kitrt_finalize();
  __kitrt_finalize();
}

int main(int argc, char *argv[]) { return 0; }
