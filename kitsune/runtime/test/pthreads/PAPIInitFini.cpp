// REQUIRES: kitsune-papi
//
// Check that PAPI is initialized and finalized correctly by Kitsune's runtime
// for the pthreads tapir target.
//
// RUN: %exe | FileCheck %s
//
// CHECK: Before __kitpthr_initialize: 0
// CHECK: After __kitpthr_initialize: {{[1-9][0-9]*}}
// CHECK: Before __kitpthr_finalize: {{[1-9][0-9]*}}
// CHECK: After __kitpthr_finalize: 0

#include "papi.h"
#include "pthreads/kitpthr.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  printf("Before __kitpthr_initialize: %d\n", PAPI_is_initialized());
  __kitpthr_initialize();
  printf("After __kitpthr_initialize: %d\n", PAPI_is_initialized());
}

__attribute__((destructor)) static void dtor(void) {
  printf("Before __kitpthr_finalize: %d\n", PAPI_is_initialized());
  __kitpthr_finalize();
  printf("After __kitpthr_finalize: %d\n", PAPI_is_initialized());
}

int main(int argc, char *argv[]) { return 0; }
