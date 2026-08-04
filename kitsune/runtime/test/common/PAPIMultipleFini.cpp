// REQUIRES: kitsune-papi
//
// Check that calling __kitpapi_finalize after it has already been finalized has
// no effect. The PAPI library should remain initialized. If the runtimes for
// multiple tapir targets are operating simultaneously, it is possible for
// __kitpapi_finalize to be called multiple times.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: PAPI initialized: {{[1-9][0-9]*}}
// CHECK: kitrt: Finalizing PAPI library
// CHECK: kitrt: Finalized PAPI library
// CHECK: PAPI initialized: 0
// CHECK: kitrt: Cannot finalize PAPI library. Not initialized
// CHECK: PAPI initialized: 0

#include "common/kitpapi.h"
#include "kitrt.h"
#include "papi.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  __kitrt_initialize();
  __kitpapi_initialize(NULL);
}

__attribute__((destructor)) static void dtor(void) {
  fprintf(stderr, "PAPI initialized: %d\n", PAPI_is_initialized());
  __kitpapi_finalize();
  fprintf(stderr, "PAPI initialized: %d\n", PAPI_is_initialized());
  __kitpapi_finalize();
  fprintf(stderr, "PAPI initialized: %d\n", PAPI_is_initialized());
  __kitrt_finalize();
}

int main(int argc, char *argv[]) { return 0; }
