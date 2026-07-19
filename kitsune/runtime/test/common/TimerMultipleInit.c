// Check that initializing the timing context multiple times has no effect. The
// context should remain initialized.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initialized = 0
// CHECK: Initializing Kitsune runtime (common)
// CHECK: Initializing Kitsune timing context
// CHECK: Initialized Kitsune timing context
// CHECK: Initialized Kitsune runtime (common)
// CHECK: Initialized = {{[1-9][0-9]*}}
// CHECK: Timing context already initialized
// CHECK: Initialized = {{[1-9][0-9]*}}

#include "common/timer.h"
#include "kitrt.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  // We need to initialize the core runtime, which will, in turn, initialize
  // the timing context. Without this, verbose mode will not be set correctly.
  fprintf(stderr, "Initialized = %d\n", __kittimer_initialized());
  __kitrt_initialize();
  fprintf(stderr, "Initialized = %d\n", __kittimer_initialized());

  // But we might as well call the initializer for the timing context directly
  // the second time around.
  __kittimer_initialize();
  fprintf(stderr, "Initialized = %d\n", __kittimer_initialized());
}

__attribute__((destructor)) static void dtor(void) { __kittimer_finalize(); }

int main(int argc, char *argv[]) { return 0; }
