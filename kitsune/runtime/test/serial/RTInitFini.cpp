// Check that the opencilk tapir target runtime initializes (and finalizes) the
// other runtime components correctly.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s --match-full-lines
//
// CHECK: kitrt: Initializing Kitsune runtime (common)
// CHECK: kitrt: Initialized Kitsune runtime (common)
// CHECK: kitrt: [serial]: Initializing Kitsune runtime (serial)
// CHECK: kitrt: [serial]: Initialized Kitsune runtime (serial)
// CHECK: Gibson
// CHECK: kitrt: [serial]: Finalizing Kitsune runtime (serial)
// CHECK: kitrt: [serial]: Finalized Kitsune runtime (serial)
// CHECK: kitrt: Finalizing Kitsune runtime (common)
// CHECK: kitrt: Finalized Kitsune runtime (common)

#include <serial/kitser.h>

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitser_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitser_finalize(); }

int main(int argc, char *argv[]) {
  fprintf(stderr, "Gibson\n");
  return 0;
}
