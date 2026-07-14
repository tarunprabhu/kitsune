// Check that the opencilk tapir target runtime initializes (and finalizes) the
// other runtime components correctly.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initializing Kitsune runtime (serial)
// CHECK: Initializing Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (serial)
// CHECK: Gibson
// CHECK: Finalizing Kitsune runtime (serial)
// CHECK: Finalizing Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (serial)

#include <serial/kitser.h>

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitser_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitser_finalize(); }

int main(int argc, char *argv[]) {
  fprintf(stderr, "Gibson\n");
  return 0;
}
