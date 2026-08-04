// REQUIRES: kitsune-papi
//
// Check that calling __kitpapi_initialize after it has already been initialized
// has no effect. The PAPI library should remain initialized. If the runtimes
// for multiple tapir targets are operating simultaneously, it is possible for
// __kitpapi_initialize to be called multiple times.
//
// RUN: env KIT_VERBOSE=1 %exe 2>&1 | FileCheck %s
//
// CHECK: PAPI initialized: 0
// CHECK: kitrt: Initializing PAPI library
// CHECK: kitrt: Initialized PAPI library
// CHECK: PAPI initialized: {{[1-9][0-9]*}}
// CHECK: kitrt: PAPI library already initialized
// CHECK: PAPI initialized: {{[1-9][0-9]*}}

#include "common/kitpapi.h"
#include "kitrt.h"
#include "papi.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  __kitrt_initialize();
  fprintf(stderr, "PAPI initialized: %d\n", PAPI_is_initialized());
  __kitpapi_initialize(NULL);
  fprintf(stderr, "PAPI initialized: %d\n", PAPI_is_initialized());
  __kitpapi_initialize(NULL);
  fprintf(stderr, "PAPI initialized: %d\n", PAPI_is_initialized());
}

__attribute__((destructor)) static void dtor(void) {
  __kitpapi_finalize();
  __kitrt_finalize();
}

int main(int argc, char *argv[]) { return 0; }
