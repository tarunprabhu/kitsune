// Check that the opencilk tapir target runtime initializes (and finalizes) the
// other runtime components correctly.
//
// RUN: KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: Initializing Kitsune runtime (opencilk)
// CHECK: Initializing Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (common)
// CHECK: Initialized Kitsune runtime (opencilk)
// CHECK: Le Guin
// CHECK: Finalizing Kitsune runtime (opencilk)
// CHECK: Finalizing Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (common)
// CHECK: Finalized Kitsune runtime (opencilk)

#include <opencilk/kitocilk.h>

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) { __kitocilk_initialize(); }

__attribute__((destructor)) static void dtor(void) { __kitocilk_finalize(); }

int main(int argc, char *argv[]) {
  fprintf(stderr, "Le Guin\n");
  return 0;
}
