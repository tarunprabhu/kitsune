// REQUIRES: kitsune-papi
//
// The initializer for the common parts of kitrt must not initialize, or
// finalize, the PAPI library.
//
// RUN: %exe | FileCheck %s
//
// CHECK: Before __kitrt_initialize: 0
// CHECK: After __kitrt_initialize: 0
// CHECK: Before __kitrt_finalize: 0
// CHECK: After __kitrt_finalize: 0

#include "kitrt.h"
#include "papi.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  printf("Before __kitrt_initialize: %d\n", PAPI_is_initialized());
  __kitrt_initialize();
  printf("After __kitrt_initialize: %d\n", PAPI_is_initialized());
}

__attribute__((destructor)) static void dtor(void) {
  printf("Before __kitrt_finalize: %d\n", PAPI_is_initialized());
  __kitrt_finalize();
  printf("After __kitrt_finalize: %d\n", PAPI_is_initialized());
}

int main(int argc, char *argv[]) { return 0; }
