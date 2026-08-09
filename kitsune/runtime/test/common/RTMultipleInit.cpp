// Check that initializing the common runtime multiple times has no effect.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initializing Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (common)
// CHECK: Kitsune runtime already initialized (common)
// CHECK-NOT: Initializing Kitsune runtime (common)

#include "kitrt.h"

#include <stdio.h>

const KitRTInitOptions initOpts{RT_NONE};

__attribute__((constructor)) static void ctor(void) {
  __kitrt_initialize(&initOpts);
  __kitrt_initialize(&initOpts);
}

__attribute__((destructor)) static void dtor(void) {
  __kitrt_finalize(&initOpts);
}

int main(int argc, char *argv[]) { return 0; }
