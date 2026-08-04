// REQUIRES: kitsune-papi
//
// Check that PAPI is initialized and finalized correctly by Kitsune's runtime
// for the opencilk tapir target.
//
// RUN: %exe | FileCheck %s
//
// CHECK: Before __kitser_initialize: 0
// CHECK: After __kitser_initialize: {{[1-9][0-9]*}}
// CHECK: Before __kitser_finalize: {{[1-9][0-9]*}}
// CHECK: After __kitser_finalize: 0

#include "papi.h"
#include "serial/kitser.h"

#include <stdio.h>

__attribute__((constructor)) static void ctor(void) {
  printf("Before __kitser_initialize: %d\n", PAPI_is_initialized());
  __kitser_initialize();
  printf("After __kitser_initialize: %d\n", PAPI_is_initialized());
}

__attribute__((destructor)) static void dtor(void) {
  printf("Before __kitser_finalize: %d\n", PAPI_is_initialized());
  __kitser_finalize();
  printf("After __kitser_finalize: %d\n", PAPI_is_initialized());
}

int main(int argc, char *argv[]) { return 0; }
