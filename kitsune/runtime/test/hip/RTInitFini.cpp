// REQUIRES: amd-gpu
//
// Check that the hip tapir target runtime initializes (and finalizes) the other
// runtime components correctly.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s --match-full-lines
//
// CHECK: kitrt: Initializing Kitsune runtime (common)
// CHECK: kitrt: Initialized Kitsune runtime (common)
// CHECK: kitrt: [hip]: Initializing Kitsune runtime (hip)
// CHECK: kitrt: [hip]: Initialized Kitsune runtime (hip)
// CHECK: Haldeman
// CHECK: kitrt: [hip]: Finalizing Kitsune runtime (hip)
// CHECK: kitrt: [hip]: Finalized Kitsune runtime (hip)
// CHECK: kitrt: Finalizing Kitsune runtime (common)
// CHECK: kitrt: Finalized Kitsune runtime (common)

#include <stdio.h>

// hip/kithip.h pulls in the hip headers. Since we don't need any of that, just
// declare what we need.
extern "C" void __kithip_initialize(void);
extern "C" void __kithip_finalize(void);

__attribute__((constructor)) static void ctor(void) { __kithip_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kithip_finalize(); }

int main(int argc, char *argv[]) {
  fprintf(stderr, "Haldeman\n");
  return 0;
}
