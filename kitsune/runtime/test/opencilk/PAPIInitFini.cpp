// REQUIRES: kitsune-papi
//
// Check that PAPI is initialized and finalized correctly by Kitsune's runtime
// for the opencilk tapir target.
//
// RUN: %exe | FileCheck %s
//
// CHECK: Before __kitocilk_initialize: 0
// CHECK: After __kitocilk_initialize: {{[1-9][0-9]*}}
// CHECK: Before __kitocilk_finalize: {{[1-9][0-9]*}}
// CHECK: After __kitocilk_finalize: 0

#include "opencilk/kitocilk.h"
#include "papi.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  printf("Before __kitocilk_initialize: %d\n", PAPI_is_initialized());
  __kitocilk_initialize();
  printf("After __kitocilk_initialize: %d\n", PAPI_is_initialized());
}

__attribute__((destructor)) static void dtor(void) {
  printf("Before __kitocilk_finalize: %d\n", PAPI_is_initialized());
  __kitocilk_finalize();
  printf("After __kitocilk_finalize: %d\n", PAPI_is_initialized());
}

int main(int argc, char *argv[]) { return 0; }
