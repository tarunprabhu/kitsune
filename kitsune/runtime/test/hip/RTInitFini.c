// REQUIRES: amd-gpu
//
// Check that the hip tapir target runtime initializes (and finalizes) the other
// runtime components correctly.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initializing Kitsune runtime (hip)
// CHECK: Initializing Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (hip)
// CHECK: Haldeman
// CHECK: Finalizing Kitsune runtime (hip)
// CHECK: Finalizing Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (hip)

#include <stdio.h>

// hip/kithip.h is not safe to be included in C source files. That header should
// be modified, but that might be a non-trivial change, so for now, this is
// easier.
void __kithip_initialize(void);
void __kithip_finalize(void);

__attribute__((constructor)) static void ctor(void) { __kithip_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kithip_finalize(); }

int main(int argc, char *argv[]) {
  fprintf(stderr, "Haldeman\n");
  return 0;
}
