// REQUIRES: kitsune-papi
//
// Check that PAPI is initialized and finalized correctly by Kitsune's runtime
// for the openmp tapir target.
//
// RUN: %exe | FileCheck %s
//
// CHECK: Before __kitomp_initialize: 0
// CHECK: After __kitomp_initialize: {{[1-9][0-9]*}}
// CHECK: Before __kitomp_finalize: {{[1-9][0-9]*}}
// CHECK: After __kitomp_finalize: 0

#include "openmp/kitomp.h"
#include "papi.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  printf("Before __kitomp_initialize: %d\n", PAPI_is_initialized());
  __kitomp_initialize();
  printf("After __kitomp_initialize: %d\n", PAPI_is_initialized());
}

__attribute__((destructor)) static void dtor(void) {
  printf("Before __kitomp_finalize: %d\n", PAPI_is_initialized());
  __kitomp_finalize();
  printf("After __kitomp_finalize: %d\n", PAPI_is_initialized());
}

int main(int argc, char *argv[]) { return 0; }
