// Check that finalizing the runtime multiple times has no effect. The runtime
// should remain finalized.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initialized = {{[1-9][0-9]*}}
// CHECK: Finalizing Kitsune timing context
// CHECK: Finalized Kitsune timing context
// CHECK: Initialized = 0
// CHECK: Cannot finalize timing context. Not initialized
// CHECK: Initialized = 0

#include "common/timer.h"
#include "kitrt.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  // We need to initialize the core runtime, which will, in turn, initialize
  // the timing context. Without this, verbose mode will not be set correctly.
  __kitrt_initialize();
}

__attribute__((destructor)) static void dtor(void) {
  fprintf(stderr, "Initialized = %d\n", __kittimer_initialized());
  __kittimer_finalize();
  fprintf(stderr, "Initialized = %d\n", __kittimer_initialized());
  __kittimer_finalize();
  fprintf(stderr, "Initialized = %d\n", __kittimer_initialized());

  // We have initialized the runtime, so be a good citizen and finalize it.
  // This will raise yet another message about the timing context not being
  // initialized, but that doesn't matter for this test.
  __kitrt_finalize();
}

int main(int argc, char *argv[]) { return 0; }
