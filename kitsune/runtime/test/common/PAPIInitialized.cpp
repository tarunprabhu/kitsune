// REQUIRES: kitsune-papi
//
// Check that __kitpapi_initialized works as expected.
//
// RUN: %exe | FileCheck %s
//
// CHECK: Before initialize: 0
// CHECK: After initialize: {{[1-9][0-9]*}}
// CHECK: Before finalize: {{[1-9][0-9]*}}
// CHECK: After finalize: 0

#include "common/kitpapi.h"
#include "kitrt.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  __kitrt_initialize();
  printf("Before initialize: %d\n", __kitpapi_initialized());
  __kitpapi_initialize(NULL);
  printf("After initialize: %d\n", __kitpapi_initialized());
}

__attribute__((destructor)) static void dtor(void) {
  printf("Before finalize: %d\n", __kitpapi_initialized());
  __kitpapi_finalize();
  printf("After finalize: %d\n", __kitpapi_initialized());
}

int main(int argc, char *argv[]) { return 0; }
