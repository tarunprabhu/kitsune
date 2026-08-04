// REQUIRES: kitsune-papi
//
// Check that PAPI is initialized and finalized correctly by Kitsune's runtime
// for the qthreads tapir target.
//
// RUN: %exe | FileCheck %s
//
// CHECK: Before __kitqthr_initialize: 0
// CHECK: After __kitqthr_initialize: {{[1-9][0-9]*}}
// CHECK: Before __kitqthr_finalize: {{[1-9][0-9]*}}
// CHECK: After __kitqthr_finalize: 0

#include "papi.h"
#include "qthreads/kitqthr.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  printf("Before __kitqthr_initialize: %d\n", PAPI_is_initialized());
  __kitqthr_initialize();
  printf("After __kitqthr_initialize: %d\n", PAPI_is_initialized());
}

__attribute__((destructor)) static void dtor(void) {
  printf("Before __kitqthr_finalize: %d\n", PAPI_is_initialized());
  __kitqthr_finalize();
  printf("After __kitqthr_finalize: %d\n", PAPI_is_initialized());
}

int main(int argc, char *argv[]) { return 0; }
