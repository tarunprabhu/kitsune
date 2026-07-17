// Check that the opencilk tapir target runtime initializes (and finalizes) the
// other runtime components correctly.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s --match-full-lines
//
// CHECK: kitrt: [opencilk]: Initializing Kitsune runtime (opencilk)
// CHECK: kitrt: Initializing Kitsune runtime (common)
// CHECK: kitrt: Initialized Kitsune runtime (common)
// CHECK: kitrt: [opencilk]: Initialized Kitsune runtime (opencilk)
// CHECK: Le Guin
// CHECK: kitrt: [opencilk]: Finalizing Kitsune runtime (opencilk)
// CHECK: kitrt: Finalizing Kitsune runtime (common)
// CHECK: kitrt: Finalized Kitsune runtime (common)
// CHECK: kitrt: [opencilk]: Finalized Kitsune runtime (opencilk)

#include <opencilk/kitocilk.h>

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitocilk_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitocilk_finalize(); }

int main(int argc, char *argv[]) {
  fprintf(stderr, "Le Guin\n");
  return 0;
}
